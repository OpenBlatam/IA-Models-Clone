"use client";

import { useCallback } from "react";
import { toast } from "sonner";

type UseJSONFormatterOptions = {
  readonly onFormat?: (formatted: string) => void;
  readonly onError?: (error: string) => void;
};

type UseJSONFormatterReturn = {
  readonly formatJSON: (value: unknown) => void;
  readonly parseJSON: (jsonString: string) => unknown | null;
};

/**
 * Custom hook for formatting and parsing JSON with error handling
 * 
 * Features:
 * - Automatic JSON formatting with proper indentation
 * - Error handling with user-friendly messages
 * - Toast notifications for success/error
 * 
 * @param options - Configuration options
 * @returns Object with format and parse functions
 */
export const useJSONFormatter = (
  options: UseJSONFormatterOptions = {}
): UseJSONFormatterReturn => {
  const { onFormat, onError } = options;

  const formatJSON = useCallback(
    (value: unknown): void => {
      try {
        const formatted = JSON.stringify(value, null, 2);
        const parsed = JSON.parse(formatted);
        onFormat?.(formatted);
        toast.success("JSON formateado correctamente");
      } catch (error) {
        const errorMessage = "No se puede formatear JSON inválido";
        onError?.(errorMessage);
        toast.error(errorMessage);
      }
    },
    [onFormat, onError]
  );

  const parseJSON = useCallback((jsonString: string): unknown | null => {
    try {
      return JSON.parse(jsonString);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "JSON inválido";
      onError?.(errorMessage);
      return null;
    }
  }, [onError]);

  return {
    formatJSON,
    parseJSON,
  };
};



