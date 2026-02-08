# Worker Tracking: Design Doc

## Problem

The Rust worker calls three SQL functions for lifecycle tracking and heartbeat:

- `ai._worker_start(text, interval) → uuid` — register on connect
- `ai._worker_heartbeat(uuid, bigint, bigint, text)` — periodic health signal
- `ai._worker_progress(uuid, int4, bigint, text)` — per-vectorizer counts

These don't exist in `setup.sql`. The worker falls back gracefully (disables heartbeat), but that means:

- No way to detect a silently dead worker
- No visibility into per-vectorizer throughput or errors
- No worker process inventory for multi-worker deployments

## What the Rust Worker Does (`worker_tracking.rs`)

1. **Feature detection:** Checks `information_schema.tables` for `ai.vectorizer_worker_process`. If missing, heartbeat is disabled entirely.
2. **Start:** Calls `ai._worker_start(version, poll_interval)`, gets back a `uuid` worker ID.
3. **Heartbeat loop:** Every `poll_interval`, calls `ai._worker_heartbeat(worker_id, successes_delta, errors_delta, last_error)`. Uses atomic counters swapped to zero each tick.
4. **Progress:** After each batch, calls `ai._worker_progress(worker_id, vectorizer_id, count, error_msg)`. `error_msg` is null on success.
5. **Shutdown:** Sends one final heartbeat with remaining counters, then exits.

Counter semantics: `successes`/`errors` in heartbeat are **deltas** accumulated since last heartbeat. The DB function should add them to running totals.

## Schema

### Tables

```sql
-- One row per worker process (alive or dead)
CREATE TABLE ai.vectorizer_worker_process (
    id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    version                     text NOT NULL,
    started                     timestamptz NOT NULL DEFAULT now(),
    expected_heartbeat_interval interval NOT NULL,
    last_heartbeat              timestamptz NOT NULL DEFAULT now(),
    heartbeat_count             bigint NOT NULL DEFAULT 0,
    success_count               bigint NOT NULL DEFAULT 0,
    error_count                 bigint NOT NULL DEFAULT 0,
    last_error_at               timestamptz,
    last_error_message          text
);

CREATE INDEX ON ai.vectorizer_worker_process (last_heartbeat);

-- Per-vectorizer progress (one row per vectorizer, shared across workers)
CREATE TABLE ai.vectorizer_worker_progress (
    vectorizer_id          int4 PRIMARY KEY NOT NULL
                           REFERENCES ai.vectorizer(id) ON DELETE CASCADE,
    success_count          bigint NOT NULL DEFAULT 0,
    error_count            bigint NOT NULL DEFAULT 0,
    last_success_at        timestamptz,
    last_success_process_id uuid,       -- no FK: see rationale below
    last_error_at          timestamptz,
    last_error_message     text,
    last_error_process_id  uuid,        -- no FK: see rationale below
    updated_at             timestamptz NOT NULL DEFAULT now()
);
```

**Why no FK from progress → process:**

1. We don't want to enforce that the process exists (the process table may be cleaned up independently).
2. We don't want any chance this row will fail to be inserted due to FK constraint.
3. We want the insert to be as fast and lightweight as possible.

**Why PK is `vectorizer_id` alone (not `(worker_id, vectorizer_id)`):**
Progress gives a single view of each vectorizer's health regardless of which worker processed it. The `last_*_process_id` columns record which worker last touched it.

### Functions

All functions use `clock_timestamp()` (not `now()`) so timestamps reflect actual wall time, not transaction start time. All are `security invoker` with `set search_path to pg_catalog, pg_temp`.

```sql
-- Register worker, return UUID
CREATE OR REPLACE FUNCTION ai._worker_start(
    version text,
    expected_heartbeat_interval interval
) RETURNS uuid AS $$
DECLARE
    worker_id uuid;
BEGIN
    INSERT INTO ai.vectorizer_worker_process (version, expected_heartbeat_interval)
    VALUES (version, expected_heartbeat_interval)
    RETURNING id INTO worker_id;
    RETURN worker_id;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER
SET search_path TO pg_catalog, pg_temp;

-- Heartbeat: bump timestamp, increment heartbeat_count, accumulate deltas
CREATE OR REPLACE FUNCTION ai._worker_heartbeat(
    worker_id uuid,
    num_successes_since_last_heartbeat bigint,
    num_errors_since_last_heartbeat bigint,
    error_message text
) RETURNS void AS $$
DECLARE
    heartbeat_timestamp timestamptz = clock_timestamp();
BEGIN
    UPDATE ai.vectorizer_worker_process SET
        last_heartbeat = heartbeat_timestamp,
        heartbeat_count = heartbeat_count + 1,
        success_count = success_count + num_successes_since_last_heartbeat,
        error_count = error_count + num_errors_since_last_heartbeat,
        last_error_message = CASE WHEN error_message IS NOT NULL
            THEN error_message ELSE last_error_message END,
        last_error_at = CASE WHEN error_message IS NOT NULL
            THEN heartbeat_timestamp ELSE last_error_at END
    WHERE id = worker_id;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER
SET search_path TO pg_catalog, pg_temp;

-- Per-vectorizer upsert: error_message NULL = success path, non-NULL = error path
CREATE OR REPLACE FUNCTION ai._worker_progress(
    worker_id uuid,
    worker_vectorizer_id int4,
    num_successes bigint,
    error_message text
) RETURNS void AS $$
DECLARE
    progress_timestamp timestamptz = clock_timestamp();
BEGIN
    INSERT INTO ai.vectorizer_worker_progress (
        vectorizer_id, success_count, error_count,
        last_success_at, last_success_process_id,
        last_error_at, last_error_message, last_error_process_id,
        updated_at
    ) VALUES (
        worker_vectorizer_id, num_successes,
        CASE WHEN error_message IS NULL THEN 0 ELSE 1 END,
        CASE WHEN error_message IS NULL THEN progress_timestamp END,
        CASE WHEN error_message IS NULL THEN worker_id END,
        CASE WHEN error_message IS NOT NULL THEN progress_timestamp END,
        error_message,
        CASE WHEN error_message IS NOT NULL THEN worker_id END,
        progress_timestamp
    )
    ON CONFLICT (vectorizer_id) DO UPDATE SET
        success_count = ai.vectorizer_worker_progress.success_count + EXCLUDED.success_count,
        error_count = ai.vectorizer_worker_progress.error_count + EXCLUDED.error_count,
        last_success_at = COALESCE(EXCLUDED.last_success_at,
            ai.vectorizer_worker_progress.last_success_at),
        last_success_process_id = COALESCE(EXCLUDED.last_success_process_id,
            ai.vectorizer_worker_progress.last_success_process_id),
        last_error_at = COALESCE(EXCLUDED.last_error_at,
            ai.vectorizer_worker_progress.last_error_at),
        last_error_message = COALESCE(EXCLUDED.last_error_message,
            ai.vectorizer_worker_progress.last_error_message),
        last_error_process_id = COALESCE(EXCLUDED.last_error_process_id,
            ai.vectorizer_worker_progress.last_error_process_id),
        updated_at = progress_timestamp;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER
SET search_path TO pg_catalog, pg_temp;
```

## Resolved Decisions

1. **`ai` schema** — consistent with existing tables, feature-detected by the worker.
2. **Index on `last_heartbeat`** — yes, enables monitoring queries like `WHERE last_heartbeat < now() - expected_heartbeat_interval * 2`.
3. **`expected_heartbeat_interval` in DB** — the worker passes its poll interval at start so monitoring can detect staleness without knowing worker config.
4. **No stale worker cleanup built-in** — each restart creates a new row. Leave cleanup to operators (simple: `DELETE FROM ai.vectorizer_worker_process WHERE last_heartbeat < now() - interval '7 days'`).
5. **`drop_vectorizer` integration** — CASCADE on the progress FK handles it automatically.

## Rust Worker Changes Needed

The Rust worker's `_worker_start` call needs to pass the `expected_heartbeat_interval` as a second argument. The current Rust code only passes `version`.
