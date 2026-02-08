pub mod merge;
pub mod split;

pub mod character;
pub mod recursive;
pub mod semchunk;
pub mod sentence;

pub use character::CharacterTextSplitter;
pub use recursive::RecursiveCharacterTextSplitter;
pub use semchunk::SemchunkSplitter;
pub use sentence::SentenceChunker;
pub use split::KeepSeparator;

/// A custom length function for text splitting (e.g. token counting).
pub type LengthFn = Box<dyn Fn(&str) -> usize + Send + Sync>;

/// Default length function: counts Unicode characters.
pub fn char_len(s: &str) -> usize {
    s.chars().count()
}
