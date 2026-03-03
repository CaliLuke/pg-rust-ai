use crate::char_len;

const DEFAULT_DELIMITERS: &[&str] = &[". ", "! ", "? ", "\n"];

/// Embedding-similarity-based chunker.
///
/// This splitter:
/// 1. splits text into sentences
/// 2. computes embeddings over sliding windows of sentences
/// 3. computes cosine similarity between adjacent windows
/// 4. smooths the curve and detects local minima as semantic boundaries
/// 5. enforces `chunk_size` with sentence-preserving fallback splitting
pub struct SemanticChunker {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub window_size: usize,
    pub skip_window: usize,
    pub reconnect_similarity_threshold: f32,
    pub max_aside_length: usize,
    pub delimiters: Vec<String>,
    pub min_characters_per_sentence: usize,
    pub strip_whitespace: bool,
    pub length_fn: Option<crate::LengthFn>,
    pub embedding_fn: Option<crate::EmbeddingFn>,
}

impl SemanticChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            window_size: 3,
            skip_window: 0,
            reconnect_similarity_threshold: 0.75,
            max_aside_length: 512,
            delimiters: DEFAULT_DELIMITERS.iter().map(|s| s.to_string()).collect(),
            min_characters_per_sentence: 12,
            strip_whitespace: true,
            length_fn: None,
            embedding_fn: None,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let default_fn = char_len;
        let len_fn: &dyn Fn(&str) -> usize = match &self.length_fn {
            Some(f) => f.as_ref(),
            None => &default_fn,
        };

        let mut sentences = split_into_sentences(text, &self.delimiters);
        sentences = merge_short_sentences(sentences, self.min_characters_per_sentence);
        if sentences.is_empty() {
            return Vec::new();
        }

        if self.window_size == 0 || sentences.len() <= self.window_size {
            return greedy_sentence_chunks(
                &sentences,
                self.chunk_size,
                self.chunk_overlap,
                self.strip_whitespace,
                len_fn,
            );
        }

        let boundaries = match &self.embedding_fn {
            Some(embed) => semantic_boundaries(&sentences, self.window_size, embed.as_ref()),
            None => Vec::new(),
        };

        let mut chunks = chunk_by_boundaries(&sentences, &boundaries, self.strip_whitespace);
        if chunks.is_empty() {
            chunks = vec![join_sentences(&sentences, self.strip_whitespace)];
        }

        if let Some(embed) = &self.embedding_fn {
            if self.skip_window > 0 {
                chunks = reconnect_skip_windows(
                    chunks,
                    self.skip_window,
                    self.reconnect_similarity_threshold,
                    self.max_aside_length,
                    embed.as_ref(),
                );
            }
        }

        let mut final_chunks: Vec<String> = Vec::new();
        for chunk in chunks {
            if len_fn(&chunk) <= self.chunk_size {
                final_chunks.push(chunk);
                continue;
            }

            // Oversized semantic chunks are split with sentence-preserving fallback.
            let chunk_sentences = split_into_sentences(&chunk, &self.delimiters);
            let sub = greedy_sentence_chunks(
                &chunk_sentences,
                self.chunk_size,
                self.chunk_overlap,
                self.strip_whitespace,
                len_fn,
            );
            final_chunks.extend(sub);
        }

        final_chunks
    }
}

fn semantic_boundaries(
    sentences: &[String],
    window_size: usize,
    embedding_fn: &dyn Fn(&str) -> Vec<f32>,
) -> Vec<usize> {
    if sentences.len() <= window_size {
        return Vec::new();
    }

    let windows: Vec<String> = (0..=sentences.len() - window_size)
        .map(|i| sentences[i..i + window_size].concat())
        .collect();

    if windows.len() < 2 {
        return Vec::new();
    }

    let embeddings: Vec<Vec<f32>> = windows.iter().map(|w| embedding_fn(w)).collect();
    let sims: Vec<f32> = embeddings
        .windows(2)
        .map(|pair| cosine_similarity(&pair[0], &pair[1]))
        .collect();

    let smoothed = savgol_smooth(&sims);
    let minima = local_minima(&smoothed);

    // Similarity index i compares windows [i..i+w) and [i+1..i+1+w).
    // We map the split around the newly introduced sentence => i + window_size.
    let mut boundaries: Vec<usize> = minima
        .into_iter()
        .map(|i| i + window_size)
        .filter(|&b| b > 0 && b < sentences.len())
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

fn chunk_by_boundaries(sentences: &[String], boundaries: &[usize], strip_whitespace: bool) -> Vec<String> {
    if sentences.is_empty() {
        return Vec::new();
    }
    if boundaries.is_empty() {
        return vec![join_sentences(sentences, strip_whitespace)];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0usize;
    for &b in boundaries {
        if b <= start || b > sentences.len() {
            continue;
        }
        let chunk = join_sentences(&sentences[start..b], strip_whitespace);
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        start = b;
    }

    if start < sentences.len() {
        let tail = join_sentences(&sentences[start..], strip_whitespace);
        if !tail.is_empty() {
            chunks.push(tail);
        }
    }

    chunks
}

fn greedy_sentence_chunks(
    sentences: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    strip_whitespace: bool,
    length_fn: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_len: usize = 0;

    for (i, sentence) in sentences.iter().enumerate() {
        let s_len = length_fn(sentence);

        if current.is_empty() {
            current.push(i);
            current_len = s_len;
            continue;
        }

        if current_len + s_len > chunk_size {
            let chunk = join_indices(sentences, &current, strip_whitespace);
            if !chunk.is_empty() {
                chunks.push(chunk);
            }

            current.clear();
            current_len = 0;

            if chunk_overlap > 0 {
                let mut overlap_len = 0usize;
                let mut overlap_start = i;
                while overlap_start > 0 {
                    let candidate = overlap_start - 1;
                    let candidate_len = length_fn(&sentences[candidate]);
                    if overlap_len + candidate_len > chunk_overlap {
                        break;
                    }
                    overlap_len += candidate_len;
                    overlap_start = candidate;
                }
                for j in overlap_start..i {
                    current.push(j);
                }
                current_len = overlap_len;
            }

            current.push(i);
            current_len += s_len;
        } else {
            current.push(i);
            current_len += s_len;
        }
    }

    if !current.is_empty() {
        let chunk = join_indices(sentences, &current, strip_whitespace);
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }

    chunks
}

fn reconnect_skip_windows(
    chunks: Vec<String>,
    skip_window: usize,
    threshold: f32,
    max_aside_length: usize,
    embedding_fn: &dyn Fn(&str) -> Vec<f32>,
) -> Vec<String> {
    if chunks.len() < 3 || skip_window == 0 {
        return chunks;
    }

    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < chunks.len() {
        let mut best_end: Option<usize> = None;
        let max_gap = skip_window.min(chunks.len().saturating_sub(i + 2));

        for gap in 1..=max_gap {
            let j = i + gap + 1;
            let aside_len: usize = chunks[i + 1..j].iter().map(|c| c.chars().count()).sum();
            if aside_len > max_aside_length {
                continue;
            }

            let left = embedding_fn(&chunks[i]);
            let right = embedding_fn(&chunks[j]);
            let sim = cosine_similarity(&left, &right);
            if sim >= threshold {
                best_end = Some(j);
            }
        }

        if let Some(end) = best_end {
            out.push(chunks[i..=end].concat());
            i = end + 1;
        } else {
            out.push(chunks[i].clone());
            i += 1;
        }
    }

    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

// Savitzky-Golay smoothing (window=5, poly=2). Falls back to identity on short input.
fn savgol_smooth(values: &[f32]) -> Vec<f32> {
    if values.len() < 5 {
        return values.to_vec();
    }
    let coeff: [f32; 5] = [-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0, 12.0 / 35.0, -3.0 / 35.0];
    let mut out = values.to_vec();
    for i in 2..values.len() - 2 {
        let mut v = 0.0f32;
        for j in 0..5 {
            v += coeff[j] * values[i + j - 2];
        }
        out[i] = v;
    }
    out
}

fn local_minima(values: &[f32]) -> Vec<usize> {
    if values.len() < 3 {
        return Vec::new();
    }
    let mut mins: Vec<usize> = Vec::new();
    for i in 1..values.len() - 1 {
        if values[i] < values[i - 1] && values[i] <= values[i + 1] {
            mins.push(i);
        }
    }
    mins
}

fn split_into_sentences(text: &str, delimiters: &[String]) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        let mut earliest_pos: Option<usize> = None;
        let mut earliest_delim_len: usize = 0;

        for delim in delimiters {
            if let Some(pos) = remaining.find(delim.as_str()) {
                match earliest_pos {
                    None => {
                        earliest_pos = Some(pos);
                        earliest_delim_len = delim.len();
                    }
                    Some(ep) => {
                        if pos < ep {
                            earliest_pos = Some(pos);
                            earliest_delim_len = delim.len();
                        }
                    }
                }
            }
        }

        match earliest_pos {
            Some(pos) => {
                let end = pos + earliest_delim_len;
                let sentence = &remaining[..end];
                if !sentence.is_empty() {
                    sentences.push(sentence.to_string());
                }
                remaining = &remaining[end..];
            }
            None => {
                if !remaining.is_empty() {
                    sentences.push(remaining.to_string());
                }
                break;
            }
        }
    }

    sentences
}

fn merge_short_sentences(sentences: Vec<String>, min_chars: usize) -> Vec<String> {
    if sentences.is_empty() {
        return sentences;
    }

    let mut result: Vec<String> = Vec::new();
    let mut buffer = String::new();

    for sentence in sentences {
        buffer.push_str(&sentence);
        if buffer.chars().count() >= min_chars {
            result.push(buffer);
            buffer = String::new();
        }
    }

    if !buffer.is_empty() {
        if let Some(last) = result.last_mut() {
            last.push_str(&buffer);
        } else {
            result.push(buffer);
        }
    }

    result
}

fn join_indices(sentences: &[String], indices: &[usize], strip_whitespace: bool) -> String {
    let joined: String = indices.iter().map(|&i| sentences[i].as_str()).collect();
    if strip_whitespace {
        joined.trim().to_string()
    } else {
        joined
    }
}

fn join_sentences(sentences: &[String], strip_whitespace: bool) -> String {
    let joined = sentences.concat();
    if strip_whitespace {
        joined.trim().to_string()
    } else {
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topic_embedding(text: &str) -> Vec<f32> {
        let lower = text.to_lowercase();
        let db = ["database", "sql", "table", "vectorizer"]
            .iter()
            .map(|k| lower.matches(k).count() as f32)
            .sum::<f32>();
        let weather = ["weather", "rain", "forecast", "temperature"]
            .iter()
            .map(|k| lower.matches(k).count() as f32)
            .sum::<f32>();
        vec![db, weather]
    }

    #[test]
    fn test_semantic_chunker_topic_boundary_split() {
        let chunker = SemanticChunker {
            chunk_size: 10_000,
            chunk_overlap: 0,
            window_size: 2,
            min_characters_per_sentence: 1,
            embedding_fn: Some(Box::new(topic_embedding)),
            ..SemanticChunker::new(10_000, 0)
        };

        let text = "SQL tables store rows. Vectorizer jobs build embeddings. Queries retrieve context. Weather forecasts predict rain. Temperature drops overnight. Storm alerts were issued.";
        let chunks = chunker.split_text(text);
        assert_eq!(chunks.len(), 2, "Expected semantic split into 2 chunks: {:?}", chunks);
        assert!(chunks[0].contains("SQL tables"));
        assert!(chunks[1].contains("Weather forecasts"));
    }

    #[test]
    fn test_semantic_chunker_enforces_chunk_size() {
        let chunker = SemanticChunker {
            chunk_size: 60,
            chunk_overlap: 0,
            window_size: 2,
            min_characters_per_sentence: 1,
            embedding_fn: Some(Box::new(topic_embedding)),
            ..SemanticChunker::new(60, 0)
        };

        let text = "SQL tables store rows for applications. Vectorizer jobs build embeddings for semantic search. Queries retrieve context from matching chunks.";
        let chunks = chunker.split_text(text);
        assert!(chunks.len() >= 2);
        for chunk in chunks {
            assert!(chunk.chars().count() <= 60, "Chunk too large: {:?}", chunk);
        }
    }

    #[test]
    fn test_semantic_chunker_without_embedding_fn_falls_back() {
        let chunker = SemanticChunker {
            chunk_size: 40,
            chunk_overlap: 0,
            window_size: 3,
            skip_window: 0,
            min_characters_per_sentence: 1,
            embedding_fn: None,
            ..SemanticChunker::new(40, 0)
        };

        let text = "Sentence one has enough text. Sentence two has enough text. Sentence three has enough text.";
        let chunks = chunker.split_text(text);
        assert!(chunks.len() >= 2, "Expected fallback size-based split");
    }

    #[test]
    fn test_reconnect_skip_windows_merges_tangential_aside() {
        let chunks = vec![
            "Database schemas and SQL indexes.".to_string(),
            "Rain and storm weather updates.".to_string(),
            "Vectorizer query planning and SQL tuning.".to_string(),
        ];
        let merged = reconnect_skip_windows(chunks, 1, 0.5, 200, &topic_embedding);
        assert_eq!(merged.len(), 1, "Expected aside reconnection merge: {:?}", merged);
        assert!(merged[0].contains("weather"));
        assert!(merged[0].contains("Vectorizer"));
    }
}
