-- Test ai.embedding_ollama() config helper

-- Default: no max_tokens
SELECT ai.embedding_ollama('nomic-embed-text', 768);

-- With max_tokens
SELECT ai.embedding_ollama('nomic-embed-text', 768, max_tokens => 2048);

-- With all params
SELECT ai.embedding_ollama(
    'nomic-embed-text', 768,
    base_url   => 'http://localhost:11434',
    options    => '{"temperature": 0.1}'::jsonb,
    keep_alive => '5m',
    max_tokens => 4096
);
