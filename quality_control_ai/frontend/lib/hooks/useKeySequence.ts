import { useEffect, useState, useRef } from 'react';

export const useKeySequence = (
  sequence: string[],
  callback: () => void,
  options: { timeout?: number } = {}
): boolean => {
  const { timeout: timeoutMs = 1000 } = options;
  const [matched, setMatched] = useState(false);
  const sequenceRef = useRef<string[]>([]);
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent): void => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      sequenceRef.current.push(event.key);
      sequenceRef.current = sequenceRef.current.slice(-sequence.length);

      if (
        sequenceRef.current.length === sequence.length &&
        sequenceRef.current.every((key, index) => key === sequence[index])
      ) {
        setMatched(true);
        callback();
        sequenceRef.current = [];
      } else {
        setMatched(false);
      }

      timeoutRef.current = setTimeout(() => {
        sequenceRef.current = [];
        setMatched(false);
      }, timeoutMs);
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [sequence, callback, timeoutMs]);

  return matched;
};

