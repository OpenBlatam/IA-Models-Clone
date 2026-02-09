"use client";

import React from "react";
import { Button } from "../ui/Button";

type FormModalFooterProps = {
  readonly onCancel: () => void;
  readonly isSubmitting: boolean;
  readonly isValid: boolean;
  readonly submitLabel?: string;
  readonly cancelLabel?: string;
};

/**
 * Shared form footer component for modals
 * 
 * Note: This component should be used inside a <form> element.
 * The submit button will trigger the form's onSubmit handler.
 */
export const FormModalFooter = ({
  onCancel,
  isSubmitting,
  isValid,
  submitLabel = "Guardar",
  cancelLabel = "Cancelar",
}: FormModalFooterProps): JSX.Element => {
  return (
    <div className="flex justify-end gap-2 pt-4 border-t">
      <Button
        type="button"
        variant="secondary"
        onClick={onCancel}
        disabled={isSubmitting}
      >
        {cancelLabel}
      </Button>
      <Button
        type="submit"
        variant="primary"
        loading={isSubmitting}
        disabled={!isValid || isSubmitting}
      >
        {submitLabel}
      </Button>
    </div>
  );
};

