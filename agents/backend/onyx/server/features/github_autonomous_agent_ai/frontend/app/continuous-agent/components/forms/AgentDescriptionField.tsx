import React, { useRef, useMemo, useCallback } from "react";
import { FormField } from "../ui/FormField";
import { Textarea } from "../ui/Textarea";
import { CharacterCount } from "../ui/CharacterCount";
import { VALIDATION_LIMITS } from "../../utils/validation";
import type { UseAgentFormReturn } from "../../hooks/useAgentForm";
import { useDebouncedCallback } from "use-debounce";

const VALIDATION_DEBOUNCE_MS = 300;

type AgentDescriptionFieldProps = {
  readonly form: UseAgentFormReturn;
  readonly inputRef?: React.RefObject<HTMLTextAreaElement>;
  readonly onErrorChange?: (error: string | null) => void;
  readonly onValidate?: () => void;
};

export const AgentDescriptionField = ({
  form,
  inputRef,
  onErrorChange,
  onValidate,
}: AgentDescriptionFieldProps): JSX.Element => {
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const ref = inputRef || internalRef;

  const debouncedValidate = useDebouncedCallback(() => {
    onValidate?.();
  }, VALIDATION_DEBOUNCE_MS);

  const descriptionCharCount = useMemo(() => form.description.length, [form.description]);

  const descriptionCharCountStatus = useMemo(() => {
    if (descriptionCharCount > VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH) {
      return "error";
    }
    return "ok";
  }, [descriptionCharCount]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>): void => {
      const value = event.target.value;
      form.setDescription(value);
      onErrorChange?.(null);

      if (form.errors.description) {
        debouncedValidate();
      } else if (value.trim().length > 0) {
        debouncedValidate();
      }
    },
    [form, debouncedValidate, onErrorChange]
  );

  return (
    <FormField
      label="Descripción"
      htmlFor="agent-description"
      required
      error={form.errors.description}
    >
      <div className="relative">
        <Textarea
          id="agent-description"
          ref={ref}
          value={form.description}
          onChange={handleChange}
          required
          rows={3}
          maxLength={VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH}
          placeholder="Describe qué hace este agente..."
          error={
            form.errors.description ||
            (descriptionCharCountStatus === "error" ? "Descripción demasiado larga" : undefined)
          }
          ariaLabel="Descripción del agente"
          ariaDescribedBy={
            form.errors.description || descriptionCharCountStatus === "error"
              ? "agent-description-error"
              : "agent-description-count"
          }
        />
        <div className="absolute right-3 top-2">
          <CharacterCount
            id="agent-description-count"
            current={descriptionCharCount}
            max={VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH}
            status={descriptionCharCountStatus}
          />
        </div>
      </div>
    </FormField>
  );
};





