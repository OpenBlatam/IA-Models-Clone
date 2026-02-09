import React, { useRef, useCallback } from "react";
import { FormField } from "../ui/FormField";
import { Input } from "../ui/Input";
import { QuickButton } from "../ui/QuickButton";
import { formatFrequency } from "../../utils/formatters";
import { FORM_DEFAULTS, FREQUENCY_EXAMPLES } from "../../constants";
import type { UseAgentFormReturn } from "../../hooks/useAgentForm";
import { useDebouncedCallback } from "use-debounce";

const VALIDATION_DEBOUNCE_MS = 300;

type AgentFrequencyFieldProps = {
  readonly form: UseAgentFormReturn;
  readonly inputRef?: React.RefObject<HTMLInputElement>;
  readonly onErrorChange?: (error: string | null) => void;
  readonly onValidate?: () => void;
};

export const AgentFrequencyField = ({
  form,
  inputRef,
  onErrorChange,
  onValidate,
}: AgentFrequencyFieldProps): JSX.Element => {
  const internalRef = useRef<HTMLInputElement>(null);
  const ref = inputRef || internalRef;

  const debouncedValidate = useDebouncedCallback(() => {
    onValidate?.();
  }, VALIDATION_DEBOUNCE_MS);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>): void => {
      const rawValue = event.target.value;
      if (rawValue === "") {
        form.setFrequency(FORM_DEFAULTS.FREQUENCY);
        return;
      }
      const value = parseInt(rawValue, 10);
      if (isNaN(value) || value < 0) {
        return;
      }
      form.setFrequency(value);
      onErrorChange?.(null);

      if (form.errors.frequency) {
        debouncedValidate();
      } else if (value >= FORM_DEFAULTS.MIN_FREQUENCY) {
        debouncedValidate();
      }
    },
    [form, debouncedValidate, onErrorChange]
  );

  const applyFrequencyPreset = useCallback(
    (seconds: number): void => {
      form.setFrequency(seconds);
      onErrorChange?.(null);
      if (ref.current) {
        ref.current.focus();
      }
    },
    [form, onErrorChange, ref]
  );

  return (
    <FormField
      label="Frecuencia (segundos)"
      htmlFor="frequency"
      required
      error={form.errors.frequency}
      helpText={`Mínimo: ${FORM_DEFAULTS.MIN_FREQUENCY} segundos. Ejemplo: ${FREQUENCY_EXAMPLES.HOUR} = 1 hora, ${FREQUENCY_EXAMPLES.DAY} = 1 día`}
    >
      <div className="space-y-2">
        <div className="flex gap-2">
          <Input
            id="frequency"
            ref={ref}
            type="number"
            value={form.frequency}
            onChange={handleChange}
            required
            min={FORM_DEFAULTS.MIN_FREQUENCY}
            step={1}
            placeholder={`${FREQUENCY_EXAMPLES.HOUR} (1 hora)`}
            error={form.errors.frequency}
            ariaLabel="Frecuencia de ejecución en segundos"
            ariaDescribedBy={form.errors.frequency ? "frequency-error" : "frequency-help"}
            className="flex-1"
          />
          <div className="flex gap-1">
            <QuickButton
              label="1h"
              onClick={() => applyFrequencyPreset(FREQUENCY_EXAMPLES.HOUR)}
              title="1 hora"
              ariaLabel="Establecer frecuencia a 1 hora"
            />
            <QuickButton
              label="1d"
              onClick={() => applyFrequencyPreset(FREQUENCY_EXAMPLES.DAY)}
              title="1 día"
              ariaLabel="Establecer frecuencia a 1 día"
            />
          </div>
        </div>
        {form.frequency >= FORM_DEFAULTS.MIN_FREQUENCY && (
          <p
            className="text-xs text-gray-500 transition-opacity duration-200"
            id="frequency-display"
            role="status"
            aria-live="polite"
          >
            Equivale a: <strong className="font-semibold">{formatFrequency(form.frequency)}</strong>
          </p>
        )}
      </div>
    </FormField>
  );
};





