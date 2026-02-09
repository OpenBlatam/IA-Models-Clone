import { useEffect, useCallback, type RefObject } from "react";

type UseFormKeyboardOptions = {
  readonly open: boolean;
  readonly isSubmitting: boolean;
  readonly isValid: boolean;
  readonly hasErrors: boolean;
  readonly onClose: () => void;
  readonly onSubmit: () => void;
  readonly onFormatJSON?: () => void;
  readonly parametersInputRef?: RefObject<HTMLElement>;
};

export const useFormKeyboard = ({
  open,
  isSubmitting,
  isValid,
  hasErrors,
  onClose,
  onSubmit,
  onFormatJSON,
  parametersInputRef,
}: UseFormKeyboardOptions): void => {
  useEffect(() => {
    if (!open) {
      return;
    }

    const handleKeyboard = (event: KeyboardEvent): void => {
      if (event.key === "Escape" && !isSubmitting) {
        onClose();
        return;
      }

      if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !isSubmitting) {
        if (isValid && !hasErrors) {
          const formElement = document.querySelector("form");
          if (formElement) {
            formElement.requestSubmit();
          }
        }
        return;
      }

      if (
        (event.ctrlKey || event.metaKey) &&
        event.key === "k" &&
        onFormatJSON &&
        parametersInputRef?.current === document.activeElement
      ) {
        event.preventDefault();
        onFormatJSON();
      }
    };

    document.addEventListener("keydown", handleKeyboard);
    return () => document.removeEventListener("keydown", handleKeyboard);
  }, [open, isSubmitting, isValid, hasErrors, onClose, onSubmit, onFormatJSON, parametersInputRef]);
};





