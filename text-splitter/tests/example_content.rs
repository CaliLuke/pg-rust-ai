use pgai_text_splitter::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, SemanticChunker, SemchunkSplitter,
    SentenceChunker,
};
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("examples")
}

fn assert_any_chunk_contains(chunks: &[String], needle: &str) {
    assert!(
        chunks.iter().any(|c| c.contains(needle)),
        "Expected at least one chunk to contain {:?}",
        needle
    );
}

#[test]
fn test_recursive_split_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = RecursiveCharacterTextSplitter::new(200, 20);
    let chunks = splitter.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 200,
            "Chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
    assert_any_chunk_contains(&chunks, "CREATE OR REPLACE FUNCTION public.summarize_article");
}

#[test]
fn test_recursive_split_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = RecursiveCharacterTextSplitter::new(500, 50);
    let chunks = splitter.split_text(&text);

    assert!(
        chunks.len() > 1,
        "610-line markdown file should produce multiple chunks, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "Chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
    assert_any_chunk_contains(&chunks, "Semantic search is a powerful feature");
}

#[test]
fn test_character_split_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    // Use a large enough chunk_size that individual lines fit
    let splitter = CharacterTextSplitter::new("\n", 500, 0);
    let chunks = splitter.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    assert!(
        chunks.len() > 1,
        "SQL file should produce multiple chunks, got {}",
        chunks.len()
    );
    assert_any_chunk_contains(&chunks, "CREATE OR REPLACE FUNCTION public.summarize_article");
}

#[test]
fn test_sentence_chunker_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let chunker = SentenceChunker::new(500, 50);
    let chunks = chunker.split_text(&text);

    assert!(
        chunks.len() > 1,
        "Markdown file should produce multiple chunks with SentenceChunker, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "SentenceChunker chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
    assert_any_chunk_contains(&chunks, "Semantic search is a powerful feature");
}

#[test]
fn test_sentence_chunker_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let chunker = SentenceChunker::new(300, 0);
    let chunks = chunker.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    assert_any_chunk_contains(&chunks, "CREATE OR REPLACE FUNCTION public.summarize_article");
}

#[test]
fn test_semchunk_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = SemchunkSplitter::new(500, 50);
    let chunks = splitter.split_text(&text);

    assert!(
        chunks.len() > 1,
        "Markdown file should produce multiple chunks with SemchunkSplitter, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "SemchunkSplitter chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
    assert_any_chunk_contains(&chunks, "Semantic search is a powerful feature");
}

#[test]
fn test_semchunk_sql_file() {
    let path = examples_dir().join("summarize_article.sql");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = SemchunkSplitter::new(200, 20);
    let chunks = splitter.split_text(&text);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 200,
            "SemchunkSplitter chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
    assert_any_chunk_contains(&chunks, "CREATE OR REPLACE FUNCTION public.summarize_article");
}

fn simple_topic_embedding(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let rag = ["rag", "retrieval", "semantic", "embedding"]
        .iter()
        .map(|k| lower.matches(k).count() as f32)
        .sum::<f32>();
    let sql = ["sql", "table", "vectorizer", "query"]
        .iter()
        .map(|k| lower.matches(k).count() as f32)
        .sum::<f32>();
    vec![rag, sql]
}

fn word_len(s: &str) -> usize {
    s.split_whitespace().count()
}

#[test]
fn test_semantic_chunker_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let chunker = SemanticChunker {
        chunk_size: 500,
        chunk_overlap: 50,
        window_size: 3,
        min_characters_per_sentence: 8,
        embedding_fn: Some(Box::new(simple_topic_embedding)),
        ..SemanticChunker::new(500, 50)
    };
    let chunks = chunker.split_text(&text);

    assert!(
        chunks.len() > 1,
        "Markdown file should produce multiple chunks with SemanticChunker, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(
            chunk.chars().count() <= 500,
            "SemanticChunker chunk exceeded chunk_size: {} chars",
            chunk.chars().count()
        );
    }
    assert_any_chunk_contains(&chunks, "Semantic search is a powerful feature");
}

#[test]
fn test_semchunk_token_aware_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let splitter = SemchunkSplitter {
        chunk_size: 90,
        chunk_overlap: 15,
        length_fn: Some(Box::new(word_len)),
        ..SemchunkSplitter::new(90, 15)
    };
    let chunks = splitter.split_text(&text);

    assert!(chunks.len() > 1, "Expected multiple token-aware chunks");
    for chunk in &chunks {
        let words = word_len(chunk);
        assert!(
            words <= 90,
            "Semchunk token-aware chunk exceeded budget: {} words",
            words
        );
    }
}

#[test]
fn test_sentence_token_aware_markdown_file() {
    let path = examples_dir().join("embeddings_from_documents/documents/pgai.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Could not read {:?}", path));

    let chunker = SentenceChunker {
        chunk_size: 90,
        chunk_overlap: 15,
        length_fn: Some(Box::new(word_len)),
        ..SentenceChunker::new(90, 15)
    };
    let chunks = chunker.split_text(&text);

    assert!(chunks.len() > 1, "Expected multiple token-aware chunks");
    for chunk in &chunks {
        let words = word_len(chunk);
        assert!(
            words <= 90,
            "Sentence token-aware chunk exceeded budget: {} words",
            words
        );
    }
}
