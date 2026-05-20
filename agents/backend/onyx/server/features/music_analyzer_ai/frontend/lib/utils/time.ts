/**
 * Time utility functions.
 * Provides helper functions for time manipulation and formatting.
 */

/**
 * Formats seconds to MM:SS format.
 * @param seconds - Seconds to format
 * @returns Formatted time string (MM:SS)
 */
export function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) {
    return '00:00';
  }

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);

  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

/**
 * Formats seconds to HH:MM:SS format.
 * @param seconds - Seconds to format
 * @returns Formatted time string (HH:MM:SS)
 */
export function formatTimeLong(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) {
    return '00:00:00';
  }

  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

/**
 * Parses a time string (MM:SS or HH:MM:SS) to seconds.
 * @param timeString - Time string to parse
 * @returns Seconds or null if invalid
 */
export function parseTime(timeString: string): number | null {
  if (!timeString || typeof timeString !== 'string') {
    return null;
  }

  const parts = timeString.split(':').map(Number);

  if (parts.length === 2) {
    // MM:SS
    const [mins, secs] = parts;
    if (isNaN(mins) || isNaN(secs)) {
      return null;
    }
    return mins * 60 + secs;
  }

  if (parts.length === 3) {
    // HH:MM:SS
    const [hours, mins, secs] = parts;
    if (isNaN(hours) || isNaN(mins) || isNaN(secs)) {
      return null;
    }
    return hours * 3600 + mins * 60 + secs;
  }

  return null;
}

/**
 * Gets the current timestamp in milliseconds.
 * @returns Current timestamp
 */
export function getTimestamp(): number {
  return Date.now();
}

/**
 * Gets the current timestamp in seconds.
 * @returns Current timestamp in seconds
 */
export function getTimestampSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

/**
 * Converts milliseconds to seconds.
 * @param ms - Milliseconds
 * @returns Seconds
 */
export function msToSeconds(ms: number): number {
  return ms / 1000;
}

/**
 * Converts seconds to milliseconds.
 * @param seconds - Seconds
 * @returns Milliseconds
 */
export function secondsToMs(seconds: number): number {
  return seconds * 1000;
}

/**
 * Gets human-readable time difference.
 * @param startTime - Start time (timestamp)
 * @param endTime - End time (timestamp, default: now)
 * @returns Human-readable time difference
 */
export function getTimeDifference(
  startTime: number,
  endTime: number = Date.now()
): string {
  const diff = Math.abs(endTime - startTime);
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days} día${days > 1 ? 's' : ''}`;
  }
  if (hours > 0) {
    return `${hours} hora${hours > 1 ? 's' : ''}`;
  }
  if (minutes > 0) {
    return `${minutes} minuto${minutes > 1 ? 's' : ''}`;
  }
  return `${seconds} segundo${seconds > 1 ? 's' : ''}`;
}

/**
 * Checks if a time is in the past.
 * @param time - Time to check (timestamp)
 * @returns True if time is in the past
 */
export function isPastTime(time: number): boolean {
  return time < Date.now();
}

/**
 * Checks if a time is in the future.
 * @param time - Time to check (timestamp)
 * @returns True if time is in the future
 */
export function isFutureTime(time: number): boolean {
  return time > Date.now();
}

