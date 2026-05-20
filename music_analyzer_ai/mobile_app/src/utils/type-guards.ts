import type { Track, TrackAnalysis, ApiError } from '../types';

export function isTrack(value: unknown): value is Track {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'name' in value &&
    'artists' in value &&
    typeof (value as Track).id === 'string' &&
    typeof (value as Track).name === 'string' &&
    Array.isArray((value as Track).artists)
  );
}

export function isTrackAnalysis(value: unknown): value is TrackAnalysis {
  return (
    typeof value === 'object' &&
    value !== null &&
    'success' in value &&
    'track_basic_info' in value &&
    'musical_analysis' in value &&
    'technical_analysis' in value &&
    typeof (value as TrackAnalysis).success === 'boolean'
  );
}

export function isApiError(value: unknown): value is ApiError {
  return (
    typeof value === 'object' &&
    value !== null &&
    'message' in value &&
    typeof (value as ApiError).message === 'string'
  );
}

export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !Number.isNaN(value);
}

export function isArray<T>(
  value: unknown,
  guard?: (item: unknown) => item is T
): value is T[] {
  if (!Array.isArray(value)) {
    return false;
  }
  if (guard) {
    return value.every(guard);
  }
  return true;
}

export function isNotNull<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

