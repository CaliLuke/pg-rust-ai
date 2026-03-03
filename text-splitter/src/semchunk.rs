use std::collections::HashMap;

use crate::char_len;

/// A semantic recursive text splitter with a fixed delimiter hierarchy.
///
/// Inspired by semchunk: uses a built-in hierarchy of 30+ delimiter types
/// ordered by semantic importance. No configuration of separators needed.
///
/// Key behaviors:
/// - **Longest-sequence-first**: prefers `\n\n\n` over `\n\n` over `\n`
/// - **Punctuation reattachment**: after splitting on non-whitespace delimiters,
///   the delimiter is reattached to the preceding chunk
/// - **Hierarchical fallback**: tries newlines, then tabs, then spaces (at
///   punctuation boundaries first), then word joiners, then individual characters
pub struct SemchunkSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub memoize: bool,
    pub strict_mode: bool,
    pub strip_whitespace: bool,
    pub length_fn: Option<crate::LengthFn>,
}

/// The hierarchy of splitter levels, ordered from most semantic to least.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitLevel {
    Newlines,            // \n sequences (longest first)
    Tabs,                // \t sequences (longest first)
    SentenceTerminators, // ". ? ! *" (space-attached first)
    ClauseSeparators,    // "; ,"
    BracketBoundaries,   // "() [] {}" (space-attached first)
    QuoteBoundaries,     // "' \"" (space-attached first)
    SentenceInterrupters, // ": -- ... …"
    SymbolBoundaries,    // "@ # $ % = + |"
    Whitespace,          // space sequences (longest first)
    WordJoiners,         // "/", "\\", "&", "-"
    Characters,          // individual chars (final fallback)
}

const SPLIT_LEVELS: &[SplitLevel] = &[
    SplitLevel::Newlines,
    SplitLevel::Tabs,
    SplitLevel::SentenceTerminators,
    SplitLevel::ClauseSeparators,
    SplitLevel::BracketBoundaries,
    SplitLevel::QuoteBoundaries,
    SplitLevel::SentenceInterrupters,
    SplitLevel::SymbolBoundaries,
    SplitLevel::Whitespace,
    SplitLevel::WordJoiners,
    SplitLevel::Characters,
];

const SENTENCE_TERMINATORS: &[&str] = &[
    "... ",
    "...",
    "… ",
    "…",
    ". ",
    "? ",
    "! ",
    "* ",
    ".",
    "?",
    "!",
    "*",
];
const STRICT_SENTENCE_TERMINATORS: &[&str] = &[
    "...\n",
    "... ",
    "...",
    "…\n",
    "… ",
    "…",
    ".\n",
    "?\n",
    "!\n",
    "*\n",
    ". ",
    "? ",
    "! ",
    "* ",
    ".",
    "?",
    "!",
    "*",
];
const CLAUSE_SEPARATORS: &[&str] = &[", ", "; ", ",", ";"];
const STRICT_CLAUSE_SEPARATORS: &[&str] = &[
    ",\n",
    ";\n",
    ", ",
    "; ",
    ",",
    ";",
];
const BRACKET_BOUNDARIES: &[&str] = &[") ", "] ", "} ", "( ", "[ ", "{ ", ")", "]", "}", "(", "[", "{"];
const STRICT_BRACKET_BOUNDARIES: &[&str] = &[
    ")\n",
    "]\n",
    "}\n",
    "( ",
    "[ ",
    "{ ",
    ") ",
    "] ",
    "} ",
    ")",
    "]",
    "}",
    "(",
    "[",
    "{",
];
const QUOTE_BOUNDARIES: &[&str] = &["\" ", "' ", "\"", "'"];
const STRICT_QUOTE_BOUNDARIES: &[&str] = &["\"\n", "'\n", "\" ", "' ", "\"", "'"];
const SENTENCE_INTERRUPTERS: &[&str] = &[" -- ", "-- ", ": ", ":", "--"];
const STRICT_SENTENCE_INTERRUPTERS: &[&str] = &["...\n", "... ", "...", " -- ", "-- ", ": ", ":\n", ":", "--"];
const SYMBOL_BOUNDARIES: &[&str] = &["@ ", "# ", "$ ", "% ", "= ", "+ ", "| ", "@", "#", "$", "%", "=", "+", "|"];
const STRICT_SYMBOL_BOUNDARIES: &[&str] = &[
    "@\n",
    "#\n",
    "$\n",
    "%\n",
    "=\n",
    "+\n",
    "|\n",
    "@ ",
    "# ",
    "$ ",
    "% ",
    "= ",
    "+ ",
    "| ",
    "@",
    "#",
    "$",
    "%",
    "=",
    "+",
    "|",
];
const WORD_JOINERS: &[&str] = &[" / ", " \\ ", " & ", " - ", "/", "\\", "&", "-"];
const STRICT_WORD_JOINERS: &[&str] = &[
    " /\n",
    " \\\n",
    " &\n",
    " -\n",
    " / ",
    " \\ ",
    " & ",
    " - ",
    "/",
    "\\",
    "&",
    "-",
];

impl SemchunkSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            memoize: true,
            strict_mode: false,
            strip_whitespace: true,
            length_fn: None,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<String> {
        let text = if self.strip_whitespace {
            text.trim()
        } else {
            text
        };
        if text.is_empty() {
            return Vec::new();
        }

        let default_fn = char_len;
        let len_fn: &dyn Fn(&str) -> usize = match &self.length_fn {
            Some(f) => f.as_ref(),
            None => &default_fn,
        };

        if len_fn(text) <= self.chunk_size {
            return vec![text.to_string()];
        }

        self.split_recursive(text, 0, len_fn)
    }

    fn split_recursive(
        &self,
        text: &str,
        level_idx: usize,
        length_fn: &dyn Fn(&str) -> usize,
    ) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        if length_fn(text) <= self.chunk_size {
            return vec![text.to_string()];
        }

        if level_idx >= SPLIT_LEVELS.len() {
            // Shouldn't happen (Characters is final fallback), but just in case
            return vec![text.to_string()];
        }

        let level = SPLIT_LEVELS[level_idx];

        if level == SplitLevel::Characters {
            // Final fallback: split into individual chars and merge
            let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            return adaptive_merge_splits(
                &chars,
                "",
                self.chunk_size,
                self.chunk_overlap,
                self.memoize,
                self.strip_whitespace,
                length_fn,
            );
        }

        // Find the best delimiter for this level that exists in the text
        let delimiter = match self.find_delimiter(text, level) {
            Some(d) => d,
            None => {
                // This level's delimiters don't exist in text, try next level
                return self.split_recursive(text, level_idx + 1, length_fn);
            }
        };

        let is_whitespace_delim = matches!(
            level,
            SplitLevel::Newlines | SplitLevel::Tabs | SplitLevel::Whitespace
        );

        // Split on the delimiter
        let raw_splits: Vec<&str> = text.split(&delimiter).collect();

        // Reattach delimiter for non-whitespace delimiters
        let splits: Vec<String> = if is_whitespace_delim {
            raw_splits
                .into_iter()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            reattach_delimiter(&raw_splits, &delimiter)
        };

        if splits.is_empty() {
            return self.split_recursive(text, level_idx + 1, length_fn);
        }

        // If splitting didn't help (only 1 piece), try next level
        if splits.len() == 1 {
            return self.split_recursive(text, level_idx + 1, length_fn);
        }

        // Merge splits, then recurse on any that still exceed chunk_size
        let merge_sep = if is_whitespace_delim { &delimiter } else { "" };

        let merged = adaptive_merge_splits(
            &splits,
            merge_sep,
            self.chunk_size,
            self.chunk_overlap,
            self.memoize,
            self.strip_whitespace,
            length_fn,
        );

        let mut result: Vec<String> = Vec::new();
        for chunk in merged {
            if length_fn(&chunk) > self.chunk_size {
                // Recurse at the next level
                let sub_chunks = self.split_recursive(&chunk, level_idx + 1, length_fn);
                result.extend(sub_chunks);
            } else {
                result.push(chunk);
            }
        }

        result
    }

    /// Find the best delimiter for a given level that exists in the text.
    /// For newlines/tabs/spaces: find the longest sequence present.
    fn find_delimiter(&self, text: &str, level: SplitLevel) -> Option<String> {
        match level {
            SplitLevel::Newlines => find_longest_sequence(text, '\n'),
            SplitLevel::Tabs => find_longest_sequence(text, '\t'),
            SplitLevel::Whitespace => find_longest_space_sequence(text),
            SplitLevel::SentenceTerminators => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_SENTENCE_TERMINATORS
                } else {
                    SENTENCE_TERMINATORS
                },
            ),
            SplitLevel::ClauseSeparators => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_CLAUSE_SEPARATORS
                } else {
                    CLAUSE_SEPARATORS
                },
            ),
            SplitLevel::BracketBoundaries => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_BRACKET_BOUNDARIES
                } else {
                    BRACKET_BOUNDARIES
                },
            ),
            SplitLevel::QuoteBoundaries => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_QUOTE_BOUNDARIES
                } else {
                    QUOTE_BOUNDARIES
                },
            ),
            SplitLevel::SentenceInterrupters => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_SENTENCE_INTERRUPTERS
                } else {
                    SENTENCE_INTERRUPTERS
                },
            ),
            SplitLevel::SymbolBoundaries => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_SYMBOL_BOUNDARIES
                } else {
                    SYMBOL_BOUNDARIES
                },
            ),
            SplitLevel::WordJoiners => find_best_delimiter(
                text,
                if self.strict_mode {
                    STRICT_WORD_JOINERS
                } else {
                    WORD_JOINERS
                },
            ),
            SplitLevel::Characters => {
                // Handled separately in split_recursive
                Some(String::new())
            }
        }
    }
}

/// Build chunks using semchunk-style adaptive search:
/// - predicts chunk span from running chars-per-unit ratio
/// - uses binary search to find the largest fitting split range
/// - memoizes length function calls (text + range caches)
fn adaptive_merge_splits(
    splits: &[String],
    separator: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    memoize: bool,
    strip_whitespace: bool,
    length_fn: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    if splits.is_empty() {
        return Vec::new();
    }

    let separator_chars = separator.chars().count();
    let separator_units = cached_len(separator, length_fn, &mut HashMap::new(), memoize);

    let mut text_len_cache: HashMap<String, usize> = HashMap::new();
    if !separator.is_empty() {
        let _ = cached_len(separator, length_fn, &mut text_len_cache, memoize);
    }

    let mut range_len_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut chunks: Vec<String> = Vec::new();
    let n = splits.len();
    let mut i: usize = 0;

    // Start conservative; update from emitted chunks.
    let mut chars_per_unit: f64 = 4.0;

    while i < n {
        let predicted_end = predict_end_by_chars(
            splits,
            i,
            n,
            separator_chars,
            chunk_size,
            chars_per_unit,
        );

        let best_end = find_max_fitting_end(
            splits,
            separator,
            i,
            predicted_end,
            n,
            chunk_size,
            memoize,
            length_fn,
            &mut text_len_cache,
            &mut range_len_cache,
        );

        let chunk = joined_range(splits, separator, i, best_end, strip_whitespace);
        if !chunk.is_empty() {
            let units = cached_len(&chunk, length_fn, &mut text_len_cache, memoize).max(1);
            let chars = chunk.chars().count();
            chars_per_unit = chars as f64 / units as f64;
            chunks.push(chunk);
        }

        if best_end >= n {
            break;
        }

        if chunk_overlap == 0 {
            i = best_end;
            continue;
        }

        // Backtrack from chunk end to build overlap-aligned start.
        let mut overlap_units: usize = 0;
        let mut next_start = best_end;
        while next_start > i {
            let idx = next_start - 1;
            let part_units = cached_len(&splits[idx], length_fn, &mut text_len_cache, memoize);
            let sep_units = if next_start < best_end {
                separator_units
            } else {
                0
            };
            let add_units = part_units + sep_units;
            if overlap_units + add_units > chunk_overlap {
                break;
            }
            overlap_units += add_units;
            next_start -= 1;
        }

        // Guarantee forward progress even when overlap budget can include all splits.
        i = next_start.max(i + 1).min(best_end);
    }

    chunks
}

fn predict_end_by_chars(
    splits: &[String],
    start: usize,
    n: usize,
    separator_chars: usize,
    chunk_size: usize,
    chars_per_unit: f64,
) -> usize {
    let char_budget = (chunk_size as f64 * chars_per_unit).max(1.0).round() as usize;
    let mut total_chars: usize = 0;
    let mut end = start;
    while end < n {
        let piece_chars = splits[end].chars().count();
        let sep_chars = if end == start { 0 } else { separator_chars };
        if total_chars + piece_chars + sep_chars > char_budget {
            break;
        }
        total_chars += piece_chars + sep_chars;
        end += 1;
    }
    end.max(start + 1)
}

fn find_max_fitting_end(
    splits: &[String],
    separator: &str,
    start: usize,
    predicted_end: usize,
    n: usize,
    chunk_size: usize,
    memoize: bool,
    length_fn: &dyn Fn(&str) -> usize,
    text_len_cache: &mut HashMap<String, usize>,
    range_len_cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    let mut low = start + 1;
    let mut high = predicted_end.min(n).max(low);

    // Ensure [low, high] brackets a transition to "too large" when possible.
    if range_len(
        splits,
        separator,
        start,
        high,
        memoize,
        length_fn,
        text_len_cache,
        range_len_cache,
    ) <= chunk_size
    {
        while high < n {
            let step = (high - start).max(1);
            let next = (high + step).min(n);
            if range_len(
                splits,
                separator,
                start,
                next,
                memoize,
                length_fn,
                text_len_cache,
                range_len_cache,
            ) <= chunk_size
            {
                high = next;
                if high == n {
                    break;
                }
            } else {
                low = high;
                high = next;
                break;
            }
        }
        if high == n
            && range_len(
                splits,
                separator,
                start,
                high,
                memoize,
                length_fn,
                text_len_cache,
                range_len_cache,
            ) <= chunk_size
        {
            return high;
        }
    }

    // If even one split is too large, force progress with single split.
    if range_len(
        splits,
        separator,
        start,
        start + 1,
        memoize,
        length_fn,
        text_len_cache,
        range_len_cache,
    ) > chunk_size
    {
        return start + 1;
    }

    // Binary search for the maximal fitting end in [low, high].
    let mut left = low;
    let mut right = high;
    while left < right {
        let mid = left + (right - left).div_ceil(2);
        if range_len(
            splits,
            separator,
            start,
            mid,
            memoize,
            length_fn,
            text_len_cache,
            range_len_cache,
        ) <= chunk_size
        {
            left = mid;
        } else {
            right = mid - 1;
        }
    }
    left
}

fn range_len(
    splits: &[String],
    separator: &str,
    start: usize,
    end: usize,
    memoize: bool,
    length_fn: &dyn Fn(&str) -> usize,
    text_len_cache: &mut HashMap<String, usize>,
    range_len_cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    if memoize {
        if let Some(len) = range_len_cache.get(&(start, end)) {
            return *len;
        }
    }
    let text = joined_range(splits, separator, start, end, false);
    let len = cached_len(&text, length_fn, text_len_cache, memoize);
    if memoize {
        range_len_cache.insert((start, end), len);
    }
    len
}

fn joined_range(
    splits: &[String],
    separator: &str,
    start: usize,
    end: usize,
    strip_whitespace: bool,
) -> String {
    let joined = splits[start..end].join(separator);
    if strip_whitespace {
        joined.trim().to_string()
    } else {
        joined
    }
}

fn cached_len(
    text: &str,
    length_fn: &dyn Fn(&str) -> usize,
    text_len_cache: &mut HashMap<String, usize>,
    memoize: bool,
) -> usize {
    if !memoize {
        return length_fn(text);
    }
    if let Some(len) = text_len_cache.get(text) {
        return *len;
    }
    let len = length_fn(text);
    text_len_cache.insert(text.to_string(), len);
    len
}

/// Find the longest contiguous sequence of `ch` in `text`.
fn find_longest_sequence(text: &str, ch: char) -> Option<String> {
    let mut max_len: usize = 0;
    let mut current_len: usize = 0;

    for c in text.chars() {
        if c == ch {
            current_len += 1;
            if current_len > max_len {
                max_len = current_len;
            }
        } else {
            current_len = 0;
        }
    }

    if max_len > 0 {
        Some(std::iter::repeat_n(ch, max_len).collect())
    } else {
        None
    }
}

/// Find the longest contiguous sequence of spaces, but only if the text
/// doesn't also contain punctuation-preceded spaces (which would be handled
/// by earlier levels).
fn find_longest_space_sequence(text: &str) -> Option<String> {
    let mut max_len: usize = 0;
    let mut current_len: usize = 0;

    for c in text.chars() {
        if c == ' ' {
            current_len += 1;
            if current_len > max_len {
                max_len = current_len;
            }
        } else {
            current_len = 0;
        }
    }

    if max_len > 0 {
        Some(" ".repeat(max_len))
    } else {
        None
    }
}

/// Find a delimiter present in text with preference for:
/// 1) longest delimiter length
/// 2) earliest position
/// 3) earliest delimiter order in the provided list
fn find_best_delimiter(text: &str, delimiters: &[&str]) -> Option<String> {
    let mut best: Option<(&str, usize, usize)> = None; // (delimiter, pos, order)

    for (order, delim) in delimiters.iter().enumerate() {
        if let Some(pos) = text.find(delim) {
            match best {
                None => best = Some((delim, pos, order)),
                Some((current, current_pos, current_order)) => {
                    let better_len = delim.len() > current.len();
                    let same_len = delim.len() == current.len();
                    let better_pos = same_len && pos < current_pos;
                    let better_order = same_len && pos == current_pos && order < current_order;
                    if better_len || better_pos || better_order {
                        best = Some((delim, pos, order));
                    }
                }
            }
        }
    }

    best.map(|(delim, _, _)| delim.to_string())
}

/// Reattach a non-whitespace delimiter to the end of the preceding split.
/// E.g., splitting "Hello. World" on ". " gives ["Hello", "World"],
/// and we want ["Hello. ", "World"].
fn reattach_delimiter(splits: &[&str], delimiter: &str) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();

    for (i, split) in splits.iter().enumerate() {
        if split.is_empty() && i > 0 {
            // Empty split after delimiter — skip
            continue;
        }
        if i < splits.len() - 1 {
            // Not the last split — reattach delimiter to end
            result.push(format!("{}{}", split, delimiter));
        } else if !split.is_empty() {
            // Last split — no delimiter after it
            result.push(split.to_string());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semchunk_basic() {
        let splitter = SemchunkSplitter::new(15, 0);
        let result = splitter.split_text("Hello world. How are you?");
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 15,
                "Chunk too long: {:?} ({})",
                chunk,
                chunk.chars().count()
            );
        }
    }

    #[test]
    fn test_semchunk_paragraph_boundaries() {
        let splitter = SemchunkSplitter::new(20, 0);
        let result = splitter.split_text("Para one.\n\nPara two.\n\nPara three.");
        // Should split on \n\n first
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 20,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_sentence_boundaries() {
        let splitter = SemchunkSplitter::new(30, 0);
        let text = "This is sentence one. This is sentence two. And sentence three.";
        let result = splitter.split_text(text);
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 30,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_punctuation_reattachment() {
        let splitter = SemchunkSplitter::new(25, 0);
        let result = splitter.split_text("Hello world. Goodbye world.");
        // Period should stay attached to preceding text
        for chunk in &result {
            // No chunk should start with ". " (delimiter should be at end of preceding)
            assert!(
                !chunk.starts_with(". "),
                "Delimiter should be reattached: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_longest_sequence_preference() {
        let splitter = SemchunkSplitter::new(10, 0);
        let text = "AAAAAAA\n\n\nBBBBBBB\nCCCCCCC";
        let result = splitter.split_text(text);
        // Should prefer triple newline first, splitting into ["AAAAAAA", "BBBBBBB\nCCCCCCC"]
        // Then "BBBBBBB\nCCCCCCC" (15 chars) > 10, recurse and split on single \n
        assert!(result.len() >= 2);
        // First chunk should be "AAAAAAA" (split on \n\n\n)
        assert_eq!(result[0], "AAAAAAA");
    }

    #[test]
    fn test_semchunk_fallback_to_characters() {
        let splitter = SemchunkSplitter::new(5, 0);
        let text = "abcdefghij";
        let result = splitter.split_text(text);
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(
                chunk.chars().count() <= 5,
                "Chunk too long: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_semchunk_empty_text() {
        let splitter = SemchunkSplitter::new(100, 0);
        let result = splitter.split_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_semchunk_whitespace_only() {
        let splitter = SemchunkSplitter::new(100, 0);
        let result = splitter.split_text("   \n\n   ");
        assert!(result.is_empty());
    }

    #[test]
    fn test_semchunk_fits_in_one_chunk() {
        let splitter = SemchunkSplitter::new(100, 0);
        let result = splitter.split_text("Short text");
        assert_eq!(result, vec!["Short text"]);
    }

    #[test]
    fn test_find_longest_sequence() {
        assert_eq!(
            find_longest_sequence("a\n\n\nb\nc", '\n'),
            Some("\n\n\n".to_string())
        );
        assert_eq!(find_longest_sequence("abc", '\n'), None);
        assert_eq!(
            find_longest_sequence("\n", '\n'),
            Some("\n".to_string())
        );
    }

    #[test]
    fn test_reattach_delimiter() {
        let splits = vec!["Hello", "World", "Foo"];
        let result = reattach_delimiter(&splits, ". ");
        assert_eq!(result, vec!["Hello. ", "World. ", "Foo"]);
    }

    #[test]
    fn test_semchunk_overlap() {
        let splitter = SemchunkSplitter::new(60, 35);
        let text = "Schemas define structure. Vectorizers create embeddings. Workers process pending rows. Queries retrieve semantic context.";
        let result = splitter.split_text(text);
        assert!(result.len() >= 2, "Expected multiple chunks, got {:?}", result);
        assert!(
            result[0].contains("Schemas define structure.")
                && result[0].contains("Vectorizers create embeddings."),
            "Expected first chunk to preserve full sentence boundaries: {:?}",
            result
        );
    }

    #[test]
    fn test_find_best_delimiter_prefers_longest_delimiter() {
        let text = "Alpha... Beta: Gamma";
        let delim = find_best_delimiter(text, &[". ", "...", ":"]);
        assert_eq!(delim, Some("...".to_string()));
    }

    #[test]
    fn test_semchunk_bracket_and_quote_boundaries() {
        let splitter = SemchunkSplitter::new(18, 0);
        let text = "Alpha (beta) [gamma] {delta} \"epsilon\" 'zeta'";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2, "Expected multiple chunks, got {:?}", chunks);
        assert!(
            chunks.iter().all(|c| c.chars().count() <= 18),
            "Chunk exceeded size: {:?}",
            chunks
        );
    }

    #[test]
    fn test_semchunk_symbol_boundaries() {
        let splitter = SemchunkSplitter::new(12, 0);
        let text = "alpha=value+delta|gamma";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2, "Expected multiple chunks, got {:?}", chunks);
        assert!(
            chunks.iter().all(|c| c.chars().count() <= 12),
            "Chunk exceeded size: {:?}",
            chunks
        );
    }

    #[test]
    fn test_semchunk_without_memoization() {
        let splitter = SemchunkSplitter {
            memoize: false,
            ..SemchunkSplitter::new(20, 5)
        };
        let text = "This is sentence one. This is sentence two. This is sentence three.";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
        for chunk in chunks {
            assert!(chunk.chars().count() <= 20, "Chunk exceeded size: {:?}", chunk);
        }
    }

    #[test]
    fn test_semchunk_strict_mode_prefers_newline_aware_delimiters() {
        let default = SemchunkSplitter {
            strict_mode: false,
            ..SemchunkSplitter::new(25, 0)
        };
        let strict = SemchunkSplitter {
            strict_mode: true,
            ..SemchunkSplitter::new(25, 0)
        };
        let text = "Alpha.\nBeta. Gamma.";

        let default_chunks = default.split_text(text);
        let strict_chunks = strict.split_text(text);

        assert!(strict_chunks.len() >= default_chunks.len());
        assert!(
            strict_chunks[0].contains("Alpha."),
            "Strict mode should preserve first sentence boundary: {:?}",
            strict_chunks
        );
    }
}
