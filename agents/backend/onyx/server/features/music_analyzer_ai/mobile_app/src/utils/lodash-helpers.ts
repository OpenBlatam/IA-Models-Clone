import {
  debounce as lodashDebounce,
  throttle as lodashThrottle,
  memoize as lodashMemoize,
  chunk,
  groupBy,
  orderBy,
  uniq,
  uniqBy,
} from 'lodash';

export { lodashDebounce, lodashThrottle, lodashMemoize, chunk, groupBy, orderBy, uniq, uniqBy };

export function groupTracksByArtist<T extends { artists: string[] }>(
  tracks: T[]
): Record<string, T[]> {
  return groupBy(tracks, (track) => track.artists[0] || 'Unknown');
}

export function sortTracksByPopularity<T extends { popularity: number }>(
  tracks: T[],
  order: 'asc' | 'desc' = 'desc'
): T[] {
  return orderBy(tracks, 'popularity', order);
}

export function getUniqueArtists(tracks: Array<{ artists: string[] }>): string[] {
  const allArtists = tracks.flatMap((track) => track.artists);
  return uniq(allArtists);
}

export function removeDuplicateTracks<T extends { id: string }>(
  tracks: T[]
): T[] {
  return uniqBy(tracks, 'id');
}

