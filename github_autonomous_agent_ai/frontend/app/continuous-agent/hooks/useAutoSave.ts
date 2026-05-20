/**
 * Custom hook for auto-saving form data to localStorage
 */

import React, { useEffect, useRef, useState } from "react";

type UseAutoSaveOptions<T> = {
  readonly data: T;
  readonly key: string;
  readonly enabled?: boolean;
  readonly debounceMs?: number;
};

export const useAutoSave = <T,>({
  data,
  key,
  enabled = true,
  debounceMs = 1000,
}: UseAutoSaveOptions<T>): void => {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      try {
        localStorage.setItem(key, JSON.stringify(data));
      } catch (error) {
        console.warn("Failed to save to localStorage:", error);
      }
    }, debounceMs);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [data, key, enabled, debounceMs]);
};

export const useAutoRestore = <T,>(
  key: string,
  defaultValue: T
): T => {
  const [restored, setRestored] = useState<T>(defaultValue);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(key);
      if (saved) {
        setRestored(JSON.parse(saved) as T);
      }
    } catch (error) {
      console.warn("Failed to restore from localStorage:", error);
    }
  }, [key]);

  return restored;
};
