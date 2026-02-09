//! Subtitle format conversion module for WASM

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SubtitleEntry {
    pub index: usize,
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
}

pub struct SubtitleConverter;

impl SubtitleConverter {
    pub fn new() -> Self {
        Self
    }

    pub fn to_srt(&self, entries: &[SubtitleEntry]) -> String {
        entries
            .iter()
            .map(|entry| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    entry.index,
                    self.format_time_srt(entry.start_time),
                    self.format_time_srt(entry.end_time),
                    entry.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn to_vtt(&self, entries: &[SubtitleEntry]) -> String {
        let mut result = "WEBVTT\n\n".to_string();
        result.push_str(
            &entries
                .iter()
                .map(|entry| {
                    format!(
                        "{} --> {}\n{}\n",
                        self.format_time_vtt(entry.start_time),
                        self.format_time_vtt(entry.end_time),
                        entry.text
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
        result
    }

    pub fn to_json(&self, entries: &[SubtitleEntry]) -> String {
        serde_json::to_string_pretty(entries).unwrap_or_default()
    }

    pub fn to_txt(&self, entries: &[SubtitleEntry]) -> String {
        entries
            .iter()
            .map(|entry| entry.text.clone())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn to_txt_with_timestamps(&self, entries: &[SubtitleEntry]) -> String {
        entries
            .iter()
            .map(|entry| {
                format!(
                    "[{}] {}",
                    self.format_time_vtt(entry.start_time),
                    entry.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn parse_srt(&self, content: &str) -> Vec<SubtitleEntry> {
        let mut entries = Vec::new();
        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() >= 3 {
                if let Ok(index) = lines[0].trim().parse::<usize>() {
                    let times: Vec<&str> = lines[1].split("-->").collect();
                    if times.len() == 2 {
                        let start = self.parse_time_srt(times[0].trim());
                        let end = self.parse_time_srt(times[1].trim());
                        let text = lines[2..].join("\n");
                        entries.push(SubtitleEntry {
                            index,
                            start_time: start,
                            end_time: end,
                            text,
                        });
                    }
                }
            }
        }

        entries
    }

    pub fn parse_vtt(&self, content: &str) -> Vec<SubtitleEntry> {
        let mut entries = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut index = 1;
        let mut i = 0;

        while i < lines.len() {
            if lines[i].contains("-->") {
                let times: Vec<&str> = lines[i].split("-->").collect();
                if times.len() == 2 {
                    let start = self.parse_time_vtt(times[0].trim());
                    let end = self.parse_time_vtt(times[1].trim());

                    let mut text_lines = Vec::new();
                    i += 1;
                    while i < lines.len() && !lines[i].is_empty() && !lines[i].contains("-->") {
                        text_lines.push(lines[i]);
                        i += 1;
                    }

                    entries.push(SubtitleEntry {
                        index,
                        start_time: start,
                        end_time: end,
                        text: text_lines.join("\n"),
                    });
                    index += 1;
                }
            }
            i += 1;
        }

        entries
    }

    pub fn shift_times(&self, entries: &[SubtitleEntry], offset_seconds: f64) -> Vec<SubtitleEntry> {
        entries
            .iter()
            .map(|entry| SubtitleEntry {
                index: entry.index,
                start_time: (entry.start_time + offset_seconds).max(0.0),
                end_time: (entry.end_time + offset_seconds).max(0.0),
                text: entry.text.clone(),
            })
            .collect()
    }

    pub fn scale_times(&self, entries: &[SubtitleEntry], factor: f64) -> Vec<SubtitleEntry> {
        entries
            .iter()
            .map(|entry| SubtitleEntry {
                index: entry.index,
                start_time: entry.start_time * factor,
                end_time: entry.end_time * factor,
                text: entry.text.clone(),
            })
            .collect()
    }

    pub fn merge_entries(&self, entries: &[SubtitleEntry], max_gap_seconds: f64) -> Vec<SubtitleEntry> {
        if entries.is_empty() {
            return Vec::new();
        }

        let mut merged = Vec::new();
        let mut current = entries[0].clone();

        for entry in entries.iter().skip(1) {
            if entry.start_time - current.end_time <= max_gap_seconds {
                current.end_time = entry.end_time;
                current.text = format!("{} {}", current.text, entry.text);
            } else {
                merged.push(current);
                current = entry.clone();
            }
        }
        merged.push(current);

        for (i, entry) in merged.iter_mut().enumerate() {
            entry.index = i + 1;
        }

        merged
    }

    fn format_time_srt(&self, seconds: f64) -> String {
        let hours = (seconds / 3600.0) as u32;
        let mins = ((seconds % 3600.0) / 60.0) as u32;
        let secs = (seconds % 60.0) as u32;
        let ms = ((seconds - seconds.floor()) * 1000.0) as u32;
        format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, ms)
    }

    fn format_time_vtt(&self, seconds: f64) -> String {
        let hours = (seconds / 3600.0) as u32;
        let mins = ((seconds % 3600.0) / 60.0) as u32;
        let secs = (seconds % 60.0) as u32;
        let ms = ((seconds - seconds.floor()) * 1000.0) as u32;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
    }

    fn parse_time_srt(&self, time_str: &str) -> f64 {
        let time_str = time_str.replace(',', ".");
        self.parse_time_vtt(&time_str)
    }

    fn parse_time_vtt(&self, time_str: &str) -> f64 {
        let parts: Vec<&str> = time_str.split(':').collect();
        match parts.len() {
            2 => {
                let mins: f64 = parts[0].parse().unwrap_or(0.0);
                let secs: f64 = parts[1].parse().unwrap_or(0.0);
                mins * 60.0 + secs
            }
            3 => {
                let hours: f64 = parts[0].parse().unwrap_or(0.0);
                let mins: f64 = parts[1].parse().unwrap_or(0.0);
                let secs: f64 = parts[2].parse().unwrap_or(0.0);
                hours * 3600.0 + mins * 60.0 + secs
            }
            _ => 0.0,
        }
    }
}

impl Default for SubtitleConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_srt() {
        let converter = SubtitleConverter::new();
        let entries = vec![SubtitleEntry {
            index: 1,
            start_time: 0.0,
            end_time: 5.0,
            text: "Hello".to_string(),
        }];
        let srt = converter.to_srt(&entries);
        assert!(srt.contains("00:00:00,000 --> 00:00:05,000"));
    }

    #[test]
    fn test_parse_srt() {
        let converter = SubtitleConverter::new();
        let srt = "1\n00:00:00,000 --> 00:00:05,000\nHello\n\n";
        let entries = converter.parse_srt(srt);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "Hello");
    }
}












