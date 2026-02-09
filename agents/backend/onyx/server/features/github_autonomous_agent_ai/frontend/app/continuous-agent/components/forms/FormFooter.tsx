import React from "react";
import { Button } from "../ui/Button";
import { UI_MESSAGES } from "../../constants/messages";

type FormFooterProps = {
  readonly isValid: boolean;
  readonly hasErrors: boolean;
  readonly isSubmitting: boolean;
  readonly onCancel: () => void;
  readonly submitLabel?: string;
};

export const FormFooter = ({
  isValid,
  hasErrors,
  isSubmitting,
  onCancel,
  submitLabel,
}: FormFooterProps): JSX.Element => {
  return (
    <div className="space-y-3 pt-4 border-t border-gray-200">
      <div className="flex gap-3">
        <Button
          type="button"
          variant="secondary"
          fullWidth
          onClick={onCancel}
          disabled={isSubmitting}
          ariaLabel="Cancelar creación de agente"
        >
          {UI_MESSAGES.CANCEL}
        </Button>
        <Button
          type="submit"
          variant="primary"
          fullWidth
          loading={isSubmitting}
          disabled={!isValid || isSubmitting || hasErrors}
          ariaLabel={submitLabel || "Crear nuevo agente"}
          aria-describedby={!isValid ? "submit-help" : undefined}
        >
          {submitLabel || UI_MESSAGES.CREATE_AGENT}
        </Button>
      </div>
      {!isValid && (
        <p
          id="submit-help"
          className="text-sm text-gray-500 text-center"
          role="status"
          aria-live="polite"
        >
          Completa todos los campos requeridos para continuar
        </p>
      )}
      <div className="flex items-center justify-center gap-4 text-xs text-gray-400 pt-2 border-t border-gray-100 flex-wrap">
        <span className="flex items-center gap-1">
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl</kbd>
          <span>+</span>
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-xs font-mono">Enter</kbd>
          <span className="ml-1">para enviar</span>
        </span>
        <span className="flex items-center gap-1">
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl</kbd>
          <span>+</span>
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-xs font-mono">K</kbd>
          <span className="ml-1">para formatear JSON</span>
        </span>
      </div>
    </div>
  );
};





