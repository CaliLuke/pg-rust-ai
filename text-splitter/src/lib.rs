pub mod merge;
pub mod split;

pub mod character;
pub mod recursive;
pub mod semchunk;
pub mod semantic;
pub mod sentence;

pub use character::CharacterTextSplitter;
pub use recursive::RecursiveCharacterTextSplitter;
pub use semchunk::SemchunkSplitter;
pub use semantic::SemanticChunker;
pub use sentence::SentenceChunker;
pub use split::KeepSeparator;

/// A custom length function for text splitting (e.g. token counting).
pub type LengthFn = Box<dyn Fn(&str) -> usize + Send + Sync>;
pub type EmbeddingFn = Box<dyn Fn(&str) -> Vec<f32> + Send + Sync>;

/// Default length function: counts Unicode characters.
pub fn char_len(s: &str) -> usize {
    s.chars().count()
}
