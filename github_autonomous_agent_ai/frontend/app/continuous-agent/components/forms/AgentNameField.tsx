import React, { useRef, useMemo, useCallback, memo } from "react";
import { FormField } from "../ui/FormField";
import { Input } from "../ui/Input";
import { CharacterCount } from "../ui/CharacterCount";
import { VALIDATION_LIMITS } from "../../utils/validation";
import type { UseAgentFormReturn } from "../../hooks/useAgentForm";
import { useDebouncedCallback } from "use-debounce";

const VALIDATION_DEBOUNCE_MS = 300;

type AgentNameFieldProps = {
  readonly form: UseAgentFormReturn;
  readonly inputRef?: React.RefObject<HTMLInputElement>;
  readonly onErrorChange?: (error: string | null) => void;
  readonly onValidate?: () => void;
};

/**
 * Form field component for agent name input
 * 
 * Features:
 * - Real-time validation with debouncing
 * - Character count indicator
 * - Accessibility support
 * 
 * @param props - Component props
 * @returns The rendered name field component
 */
const AgentNameFieldComponent = ({
  form,
  inputRef,
  onErrorChange,
  onValidate,
}: AgentNameFieldProps): JSX.Element => {
  const internalRef = useRef<HTMLInputElement>(null);
  const ref = inputRef || internalRef;

  const debouncedValidate = useDebouncedCallback(() => {
    onValidate?.();
  }, VALIDATION_DEBOUNCE_MS);

  const nameCharCount = useMemo(() => form.name.length, [form.name]);

  const nameCharCountStatus = useMemo(() => {
    if (nameCharCount < VALIDATION_LIMITS.MIN_NAME_LENGTH) {
      return "warning";
    }
    if (nameCharCount > VALIDATION_LIMITS.MAX_NAME_LENGTH) {
      return "error";
    }
    return "ok";
  }, [nameCharCount]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>): void => {
      const value = event.target.value;
      form.setName(value);
      onErrorChange?.(null);

      if (form.errors.name) {
        debouncedValidate();
      } else if (value.trim().length >= VALIDATION_LIMITS.MIN_NAME_LENGTH) {
        debouncedValidate();
      }
    },
    [form, debouncedValidate, onErrorChange]
  );

  return (
    <FormField label="Nombre del Agente" htmlFor="agent-name" required error={form.errors.name}>
      <div className="relative">
        <Input
          id="agent-name"
          ref={ref}
          type="text"
          value={form.name}
          onChange={handleChange}
          required
          maxLength={VALIDATION_LIMITS.MAX_NAME_LENGTH}
          placeholder="Ej: Generador de contenido automático"
          error={
            form.errors.name ||
            (nameCharCountStatus === "error" ? "Nombre demasiado largo" : undefined)
          }
          ariaLabel="Nombre del agente"
          ariaDescribedBy={
            form.errors.name || nameCharCountStatus === "error"
              ? "agent-name-error"
              : "agent-name-count"
          }
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <CharacterCount
            id="agent-name-count"
            current={nameCharCount}
            max={VALIDATION_LIMITS.MAX_NAME_LENGTH}
            min={VALIDATION_LIMITS.MIN_NAME_LENGTH}
            status={nameCharCountStatus}
          />
        </div>
      </div>
      {nameCharCount < VALIDATION_LIMITS.MIN_NAME_LENGTH && !form.errors.name && (
        <p className="text-xs text-yellow-600 mt-1" role="status">
          Mínimo {VALIDATION_LIMITS.MIN_NAME_LENGTH} caracteres
        </p>
      )}
    </FormField>
  );
};

// Memoize component to prevent unnecessary re-renders
export const AgentNameField = memo(AgentNameFieldComponent, (prevProps, nextProps) => {
  return (
    prevProps.form.name === nextProps.form.name &&
    prevProps.form.errors.name === nextProps.form.errors.name &&
    prevProps.inputRef === nextProps.inputRef &&
    prevProps.onErrorChange === nextProps.onErrorChange &&
    prevProps.onValidate === nextProps.onValidate
  );
});

AgentNameField.displayName = "AgentNameField";


