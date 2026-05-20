import { useState, useCallback } from "react";
import { toast } from "sonner";
import { logError } from "../utils/logger";

type UseCopyToClipboardOptions = {
  readonly successMessage?: string;
  readonly errorMessage?: string;
  readonly resetDelay?: number;
};

type UseCopyToClipboardReturn = {
  readonly isCopied: boolean;
  readonly copy: (text: string) => Promise<void>;
  readonly reset: () => void;
};

/**
 * Hook for copying text to clipboard with feedback
 * 
 * Features:
 * - Automatic success/error notifications
 * - Visual feedback state
 * - Error logging
 * - Configurable messages
 */
export const useCopyToClipboard = (
  options: UseCopyToClipboardOptions = {}
): UseCopyToClipboardReturn => {
  const {
    successMessage = "Copiado al portapapeles",
    errorMessage = "Error al copiar al portapapeles",
    resetDelay = 2000,
  } = options;

  const [isCopied, setIsCopied] = useState(false);

  const copy = useCallback(
    async (text: string): Promise<void> => {
      try {
        await navigator.clipboard.writeText(text);
        setIsCopied(true);
        toast.success(successMessage);
        setTimeout(() => setIsCopied(false), resetDelay);
      } catch (error) {
        logError("Failed to copy to clipboard", error instanceof Error ? error : new Error(String(error)));
        toast.error(errorMessage);
      }
    },
    [successMessage, errorMessage, resetDelay]
  );

  const reset = useCallback((): void => {
    setIsCopied(false);
  }, []);

  return {
    isCopied,
    copy,
    reset,
  };
};



