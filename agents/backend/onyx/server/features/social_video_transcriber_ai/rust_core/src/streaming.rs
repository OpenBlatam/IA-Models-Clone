//! Streaming Text Processing
//!
//! Provides efficient streaming processing for large text:
//! - Chunked text processing
//! - Incremental analysis
//! - Memory-efficient iteration
//! - Progress tracking

use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use parking_lot::RwLock;
use rayon::prelude::*;

#[pyclass]
pub struct TextStream {
    text: String,
    chunk_size: usize,
    current_pos: AtomicUsize,
    overlap: usize,
}

#[pymethods]
impl TextStream {
    #[new]
    #[pyo3(signature = (text, chunk_size=1000, overlap=100))]
    pub fn new(text: String, chunk_size: usize, overlap: usize) -> Self {
        Self {
            text,
            chunk_size,
            current_pos: AtomicUsize::new(0),
            overlap,
        }
    }

    pub fn next_chunk(&self) -> Option<StreamChunk> {
        let pos = self.current_pos.load(Ordering::Acquire);
        if pos >= self.text.len() {
            return None;
        }

        let end = (pos + self.chunk_size).min(self.text.len());
        let chunk_text = &self.text[pos..end];

        let chunk = StreamChunk {
            index: pos / self.chunk_size.max(1),
            start: pos,
            end,
            text: chunk_text.to_string(),
            is_last: end >= self.text.len(),
        };

        let next_pos = if end >= self.text.len() {
            self.text.len()
        } else {
            end.saturating_sub(self.overlap)
        };
        self.current_pos.store(next_pos, Ordering::Release);

        Some(chunk)
    }

    pub fn reset(&self) {
        self.current_pos.store(0, Ordering::Release);
    }

    pub fn progress(&self) -> f64 {
        let pos = self.current_pos.load(Ordering::Acquire);
        if self.text.is_empty() {
            1.0
        } else {
            pos as f64 / self.text.len() as f64
        }
    }

    pub fn remaining(&self) -> usize {
        let pos = self.current_pos.load(Ordering::Acquire);
        self.text.len().saturating_sub(pos)
    }

    pub fn total_chunks(&self) -> usize {
        if self.text.is_empty() || self.chunk_size == 0 {
            0
        } else {
            let effective_chunk = self.chunk_size.saturating_sub(self.overlap).max(1);
            (self.text.len() + effective_chunk - 1) / effective_chunk
        }
    }

    pub fn collect_all(&self) -> Vec<StreamChunk> {
        self.reset();
        let mut chunks = Vec::new();
        while let Some(chunk) = self.next_chunk() {
            chunks.push(chunk);
        }
        chunks
    }
}

#[pyclass]
#[derive(Clone)]
pub struct StreamChunk {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub is_last: bool,
}

#[pymethods]
impl StreamChunk {
    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamChunk(index={}, start={}, end={}, len={}, is_last={})",
            self.index,
            self.start,
            self.end,
            self.len(),
            self.is_last
        )
    }
}

#[pyclass]
pub struct ParallelProcessor {
    num_workers: usize,
    cancelled: AtomicBool,
    processed: AtomicUsize,
    total: AtomicUsize,
}

#[pymethods]
impl ParallelProcessor {
    #[new]
    #[pyo3(signature = (num_workers=None))]
    pub fn new(num_workers: Option<usize>) -> Self {
        let workers = num_workers.unwrap_or_else(num_cpus::get);
        Self {
            num_workers: workers,
            cancelled: AtomicBool::new(false),
            processed: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
        }
    }

    pub fn process_chunks(&self, chunks: Vec<String>) -> Vec<ChunkResult> {
        self.processed.store(0, Ordering::Release);
        self.total.store(chunks.len(), Ordering::Release);
        self.cancelled.store(false, Ordering::Release);

        chunks
            .par_iter()
            .enumerate()
            .filter_map(|(idx, chunk)| {
                if self.cancelled.load(Ordering::Acquire) {
                    return None;
                }

                let result = self.process_single(idx, chunk);
                self.processed.fetch_add(1, Ordering::AcqRel);
                Some(result)
            })
            .collect()
    }

    fn process_single(&self, index: usize, chunk: &str) -> ChunkResult {
        let word_count = chunk.split_whitespace().count();
        let char_count = chunk.chars().count();
        let line_count = chunk.lines().count();

        ChunkResult {
            index,
            word_count,
            char_count,
            line_count,
            success: true,
            error: None,
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn progress(&self) -> ProcessingProgress {
        let processed = self.processed.load(Ordering::Acquire);
        let total = self.total.load(Ordering::Acquire);

        ProcessingProgress {
            processed,
            total,
            percentage: if total == 0 {
                100.0
            } else {
                processed as f64 / total as f64 * 100.0
            },
            cancelled: self.cancelled.load(Ordering::Acquire),
        }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ChunkResult {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub word_count: usize,
    #[pyo3(get)]
    pub char_count: usize,
    #[pyo3(get)]
    pub line_count: usize,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl ChunkResult {
    fn __repr__(&self) -> String {
        format!(
            "ChunkResult(idx={}, words={}, chars={}, success={})",
            self.index, self.word_count, self.char_count, self.success
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ProcessingProgress {
    #[pyo3(get)]
    pub processed: usize,
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub percentage: f64,
    #[pyo3(get)]
    pub cancelled: bool,
}

#[pymethods]
impl ProcessingProgress {
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.processed)
    }

    pub fn is_complete(&self) -> bool {
        self.processed >= self.total
    }

    fn __repr__(&self) -> String {
        format!(
            "Progress({}/{} = {:.1}%{})",
            self.processed,
            self.total,
            self.percentage,
            if self.cancelled { " CANCELLED" } else { "" }
        )
    }
}

#[pyclass]
pub struct LineIterator {
    lines: Vec<String>,
    current: AtomicUsize,
}

#[pymethods]
impl LineIterator {
    #[new]
    pub fn new(text: String) -> Self {
        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        Self {
            lines,
            current: AtomicUsize::new(0),
        }
    }

    pub fn next_line(&self) -> Option<LineItem> {
        let idx = self.current.fetch_add(1, Ordering::AcqRel);
        if idx >= self.lines.len() {
            return None;
        }

        Some(LineItem {
            index: idx,
            text: self.lines[idx].clone(),
            is_last: idx == self.lines.len() - 1,
        })
    }

    pub fn reset(&self) {
        self.current.store(0, Ordering::Release);
    }

    pub fn total_lines(&self) -> usize {
        self.lines.len()
    }

    pub fn remaining(&self) -> usize {
        let current = self.current.load(Ordering::Acquire);
        self.lines.len().saturating_sub(current)
    }

    pub fn skip(&self, n: usize) {
        self.current.fetch_add(n, Ordering::AcqRel);
    }
}

#[pyclass]
#[derive(Clone)]
pub struct LineItem {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub is_last: bool,
}

#[pymethods]
impl LineItem {
    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    fn __repr__(&self) -> String {
        format!("Line({}): {}", self.index, self.text.chars().take(50).collect::<String>())
    }
}

#[pyfunction]
pub fn create_text_stream(text: String, chunk_size: usize) -> TextStream {
    TextStream::new(text, chunk_size, chunk_size / 10)
}

#[pyfunction]
pub fn create_line_iterator(text: String) -> LineIterator {
    LineIterator::new(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_stream() {
        let stream = TextStream::new("hello world test".to_string(), 5, 1);
        
        let chunk1 = stream.next_chunk().unwrap();
        assert_eq!(chunk1.text, "hello");
        assert_eq!(chunk1.index, 0);

        let chunk2 = stream.next_chunk().unwrap();
        assert!(!chunk2.text.is_empty());
    }

    #[test]
    fn test_parallel_processor() {
        let processor = ParallelProcessor::new(Some(2));
        let chunks = vec!["hello world".to_string(), "test chunk".to_string()];
        
        let results = processor.process_chunks(chunks);
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
    }

    #[test]
    fn test_line_iterator() {
        let iter = LineIterator::new("line1\nline2\nline3".to_string());
        
        let line1 = iter.next_line().unwrap();
        assert_eq!(line1.text, "line1");
        assert_eq!(line1.index, 0);

        let line2 = iter.next_line().unwrap();
        assert_eq!(line2.text, "line2");
    }
}












