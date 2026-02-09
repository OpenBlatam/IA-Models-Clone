import { useEffect } from "react";

type UseModalKeyboardShortcutsOptions = {
  readonly open: boolean;
  readonly onSubmit?: () => void;
  readonly onFormatJSON?: () => void;
  readonly canSubmit?: boolean;
};

/**
 * Hook for handling keyboard shortcuts in modals
 * 
 * Shortcuts:
 * - Ctrl/Cmd + Enter: Submit form (if canSubmit is true)
 * - Shift + F: Format JSON (if onFormatJSON is provided)
 * 
 * @param options - Configuration options
 */
export const useModalKeyboardShortcuts = ({
  open,
  onSubmit,
  onFormatJSON,
  canSubmit = false,
}: UseModalKeyboardShortcutsOptions): void => {
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.ctrlKey || event.metaKey) {
        if (event.key === "Enter" && canSubmit && onSubmit) {
          event.preventDefault();
          onSubmit();
        } else if (event.shiftKey && event.key === "F" && onFormatJSON) {
          event.preventDefault();
          onFormatJSON();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onSubmit, onFormatJSON, canSubmit]);
};



