export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

export function formatArtists(artists: string[]): string {
  return artists.join(', ');
}

export function formatBPM(bpm: number): string {
  return `${Math.round(bpm)} BPM`;
}

export function formatPercentage(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatKeySignature(
  rootNote: string,
  mode: string
): string {
  return `${rootNote} ${mode}`;
}

// Re-export date helpers for convenience
export {
  formatDate,
  formatRelativeTime,
  formatDuration as formatDurationFromDate,
  getTimeAgo,
  isRecent,
} from './date-helpers';

