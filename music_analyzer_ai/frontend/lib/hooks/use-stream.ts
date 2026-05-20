/**
 * Custom hook for stream processing.
 * Provides reactive stream functionality.
 */

import { useMemo } from 'react';
import { Stream, stream } from '../utils/stream';

/**
 * Custom hook for stream processing.
 * Provides reactive stream functionality.
 *
 * @param source - Stream source
 * @returns Stream instance
 */
export function useStream<T>(
  source: Iterator<T> | Iterable<T>
): Stream<T> {
  return useMemo(() => stream(source), [source]);
}

