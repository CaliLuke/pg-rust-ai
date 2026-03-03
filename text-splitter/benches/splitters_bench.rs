use std::path::PathBuf;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use pgai_text_splitter::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, SemanticChunker, SemchunkSplitter,
    SentenceChunker,
};
use tiktoken_rs::cl100k_base;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("examples")
}

fn read_fixture(path: &str) -> String {
    let full = examples_dir().join(path);
    std::fs::read_to_string(&full).unwrap_or_else(|_| panic!("Could not read {:?}", full))
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

fn bench_splitters(c: &mut Criterion) {
    let sql = read_fixture("summarize_article.sql");
    let md = read_fixture("embeddings_from_documents/documents/pgai.md");

    let character = CharacterTextSplitter::new("\n", 500, 50);
    let recursive = RecursiveCharacterTextSplitter::new(500, 50);
    let sentence = SentenceChunker::new(500, 50);
    let semchunk = SemchunkSplitter::new(500, 50);
    let semchunk_token_memo = SemchunkSplitter {
        chunk_size: 90,
        chunk_overlap: 15,
        memoize: true,
        length_fn: Some(Box::new(word_len)),
        ..SemchunkSplitter::new(90, 15)
    };
    let semchunk_token_no_memo = SemchunkSplitter {
        chunk_size: 90,
        chunk_overlap: 15,
        memoize: false,
        length_fn: Some(Box::new(word_len)),
        ..SemchunkSplitter::new(90, 15)
    };
    let tiktoken = Arc::new(cl100k_base().expect("cl100k tokenizer must initialize"));
    let semchunk_tiktoken_memo = {
        let bpe = tiktoken.clone();
        SemchunkSplitter {
            chunk_size: 120,
            chunk_overlap: 20,
            memoize: true,
            length_fn: Some(Box::new(move |s: &str| bpe.encode_with_special_tokens(s).len())),
            ..SemchunkSplitter::new(120, 20)
        }
    };
    let semchunk_tiktoken_no_memo = {
        let bpe = tiktoken.clone();
        SemchunkSplitter {
            chunk_size: 120,
            chunk_overlap: 20,
            memoize: false,
            length_fn: Some(Box::new(move |s: &str| bpe.encode_with_special_tokens(s).len())),
            ..SemchunkSplitter::new(120, 20)
        }
    };
    let semantic = SemanticChunker {
        chunk_size: 500,
        chunk_overlap: 50,
        window_size: 3,
        min_characters_per_sentence: 8,
        embedding_fn: Some(Box::new(simple_topic_embedding)),
        ..SemanticChunker::new(500, 50)
    };

    let mut group = c.benchmark_group("splitters_markdown");
    group.bench_function("character", |b| b.iter(|| character.split_text(&md)));
    group.bench_function("recursive", |b| b.iter(|| recursive.split_text(&md)));
    group.bench_function("sentence", |b| b.iter(|| sentence.split_text(&md)));
    group.bench_function("semchunk", |b| b.iter(|| semchunk.split_text(&md)));
    group.bench_function("semchunk_token_memo", |b| {
        b.iter(|| semchunk_token_memo.split_text(&md))
    });
    group.bench_function("semchunk_token_no_memo", |b| {
        b.iter(|| semchunk_token_no_memo.split_text(&md))
    });
    group.bench_function("semchunk_tiktoken_memo", |b| {
        b.iter(|| semchunk_tiktoken_memo.split_text(&md))
    });
    group.bench_function("semchunk_tiktoken_no_memo", |b| {
        b.iter(|| semchunk_tiktoken_no_memo.split_text(&md))
    });
    group.bench_function("semantic", |b| b.iter(|| semantic.split_text(&md)));
    group.finish();

    let mut group = c.benchmark_group("splitters_sql");
    group.bench_function("character", |b| b.iter(|| character.split_text(&sql)));
    group.bench_function("recursive", |b| b.iter(|| recursive.split_text(&sql)));
    group.bench_function("sentence", |b| b.iter(|| sentence.split_text(&sql)));
    group.bench_function("semchunk", |b| b.iter(|| semchunk.split_text(&sql)));
    group.bench_function("semchunk_token_memo", |b| {
        b.iter(|| semchunk_token_memo.split_text(&sql))
    });
    group.bench_function("semchunk_token_no_memo", |b| {
        b.iter(|| semchunk_token_no_memo.split_text(&sql))
    });
    group.bench_function("semchunk_tiktoken_memo", |b| {
        b.iter(|| semchunk_tiktoken_memo.split_text(&sql))
    });
    group.bench_function("semchunk_tiktoken_no_memo", |b| {
        b.iter(|| semchunk_tiktoken_no_memo.split_text(&sql))
    });
    group.bench_function("semantic", |b| {
        b.iter_batched(
            || sql.clone(),
            |input| semantic.split_text(&input),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_splitters);
criterion_main!(benches);
