"use client";

import React, { useCallback } from "react";
import { useFormContext, Controller } from "react-hook-form";
import type { AgentFormValues } from "../../../utils/validation/zod-schemas";
import { FormField } from "../../ui/FormField";
import { Input } from "../../ui/Input";
import { QuickButton } from "../../ui/QuickButton";
import { formatFrequency } from "../../../utils/formatters";
import { FORM_DEFAULTS, FREQUENCY_EXAMPLES } from "../../../constants";

/**
 * Frequency field component with presets and formatted display
 */
export const FrequencyField = (): JSX.Element => {
  const {
    control,
    formState: { errors },
    watch,
    setValue,
  } = useFormContext<AgentFormValues>();

  const frequencyValue = watch("config.frequency") || FORM_DEFAULTS.FREQUENCY;

  const applyPreset = useCallback(
    (seconds: number): void => {
      setValue("config.frequency", seconds, { shouldValidate: true });
    },
    [setValue]
  );

  return (
    <FormField
      label="Frecuencia (segundos)"
      htmlFor="agent-frequency"
      error={errors.config?.frequency?.message}
      helpText={`Mínimo: ${FORM_DEFAULTS.MIN_FREQUENCY} segundos. Ejemplo: ${FREQUENCY_EXAMPLES.HOUR} = 1 hora, ${FREQUENCY_EXAMPLES.DAY} = 1 día`}
    >
      <div className="space-y-2">
        <div className="flex gap-2">
          <Controller
            name="config.frequency"
            control={control}
            render={({ field }) => (
              <Input
                id="agent-frequency"
                type="number"
                {...field}
                value={field.value ?? ""}
                onChange={(e) => field.onChange(Number(e.target.value) || FORM_DEFAULTS.FREQUENCY)}
                min={FORM_DEFAULTS.MIN_FREQUENCY}
                step={1}
                placeholder={`${FREQUENCY_EXAMPLES.HOUR} (1 hora)`}
                error={errors.config?.frequency?.message}
                className="flex-1"
              />
            )}
          />
          <div className="flex gap-1">
            <QuickButton
              label="1h"
              onClick={() => applyPreset(FREQUENCY_EXAMPLES.HOUR)}
              title="1 hora"
              ariaLabel="Establecer frecuencia a 1 hora"
            />
            <QuickButton
              label="1d"
              onClick={() => applyPreset(FREQUENCY_EXAMPLES.DAY)}
              title="1 día"
              ariaLabel="Establecer frecuencia a 1 día"
            />
          </div>
        </div>
        {frequencyValue >= FORM_DEFAULTS.MIN_FREQUENCY && (
          <p
            className="text-xs text-gray-500 transition-opacity duration-200"
            role="status"
            aria-live="polite"
          >
            Equivale a: <strong className="font-semibold">{formatFrequency(frequencyValue)}</strong>
          </p>
        )}
      </div>
    </FormField>
  );
};



