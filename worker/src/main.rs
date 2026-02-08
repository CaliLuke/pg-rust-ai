use anyhow::Result;
use clap::Parser;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Database URL
    #[arg(short, long, env = "DB_URL")]
    db_url: String,

    /// Poll interval in seconds
    #[arg(short, long, default_value_t = 60)]
    poll_interval: u64,

    /// Run once and exit
    #[arg(long, default_value_t = false)]
    once: bool,

    /// Exit on error
    #[arg(long, default_value_t = false)]
    exit_on_error: bool,

    /// Specific vectorizer IDs to run (comma-separated)
    #[arg(long, value_delimiter = ',')]
    vectorizer_ids: Vec<i32>,

    /// Log output format: "text" or "json"
    #[arg(long, default_value = "text")]
    log_format: String,
}

/// Redact the password in a database connection URL.
fn redact_db_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) => {
            if parsed.password().is_some() {
                let _ = parsed.set_password(Some("***"));
            }
            parsed.to_string()
        }
        Err(_) => "***".to_string(),
    }
}

/// Initialize telemetry: fmt layer + env filter + optional OTLP traces.
/// Returns the TracerProvider if OTLP was configured (caller must shut it down).
fn init_telemetry(json: bool) -> Option<SdkTracerProvider> {
    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    let provider = otel_endpoint.map(|endpoint| {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build()
            .expect("failed to create OTLP exporter");

        SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build()
    });

    // Each combination of (json, otel) needs its own subscriber stack because
    // the concrete types differ and tracing_subscriber is generic.
    match (json, &provider) {
        (true, Some(p)) => {
            let tracer = p.tracer("pgai-worker");
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer().json())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .init();
        }
        (false, Some(p)) => {
            let tracer = p.tracer("pgai-worker");
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .init();
        }
        (true, None) => {
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer().json())
                .init();
        }
        (false, None) => {
            tracing_subscriber::registry()
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
                .with(tracing_subscriber::fmt::layer())
                .init();
        }
    }

    if provider.is_some() {
        info!("OpenTelemetry tracing enabled");
    }

    provider
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let args = Args::parse();
    let tracer_provider = init_telemetry(args.log_format == "json");

    let cancel = CancellationToken::new();

    // Spawn signal handler
    let cancel_for_signal = cancel.clone();
    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => {
                    info!("Received SIGINT, initiating graceful shutdown");
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                }
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.expect("failed to listen for ctrl_c");
            info!("Received Ctrl+C, initiating graceful shutdown");
        }
        cancel_for_signal.cancel();
    });

    info!("Starting pgai worker-rs");

    info!(
        db_url = %redact_db_url(&args.db_url),
        poll_interval_secs = args.poll_interval,
        once = args.once,
        exit_on_error = args.exit_on_error,
        vectorizer_ids = ?args.vectorizer_ids,
        log_format = %args.log_format,
        "Worker configuration"
    );

    let worker = worker::Worker::new(
        &args.db_url,
        Duration::from_secs(args.poll_interval),
        args.once,
        args.vectorizer_ids,
        args.exit_on_error,
        cancel,
    )
    .await?;

    worker.run().await?;

    if let Some(provider) = tracer_provider {
        provider.shutdown().ok();
    }

    info!("Worker shut down cleanly");
    Ok(())
}
