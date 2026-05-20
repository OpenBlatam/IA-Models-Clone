"use client";

import React, { useState, useMemo } from "react";
import { useFormContext, Controller } from "react-hook-form";
import type { AgentFormValues } from "../../../utils/validation/zod-schemas";
import { FormField } from "../../ui/FormField";
import { Textarea } from "../../ui/Textarea";
import { Button } from "../../ui/Button";
import { JSONTemplates } from "../../ui/JSONTemplates";

const JSON_TEMPLATES = [
  {
    name: "Vacío",
    value: "{}",
    description: "JSON vacío",
  },
  {
    name: "Básico",
    value: JSON.stringify(
      {
        timeout: 30,
        retries: 3,
      },
      null,
      2
    ),
    description: "Configuración básica con timeout y reintentos",
  },
  {
    name: "Avanzado",
    value: JSON.stringify(
      {
        timeout: 60,
        retries: 5,
        headers: {
          "Content-Type": "application/json",
        },
        options: {
          async: true,
          cache: false,
        },
      },
      null,
      2
    ),
    description: "Configuración avanzada con headers y opciones",
  },
] as const;

type ParametersFieldProps = {
  readonly onFormatJSON: () => void;
};

/**
 * Parameters field component with JSON templates and formatting
 */
export const ParametersField = ({ onFormatJSON }: ParametersFieldProps): JSX.Element => {
  const {
    control,
    formState: { errors },
    setValue,
  } = useFormContext<AgentFormValues>();
  const [showTemplates, setShowTemplates] = useState(false);

  return (
    <FormField
      label="Parámetros (JSON)"
      htmlFor="agent-parameters"
      error={errors.config?.parameters?.message}
    >
      <div className="space-y-2">
        <Controller
          name="config.parameters"
          control={control}
          render={({ field }) => (
            <Textarea
              id="agent-parameters"
              value={JSON.stringify(field.value || {}, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  field.onChange(parsed);
                } catch {
                  field.onChange(e.target.value);
                }
              }}
              placeholder='{"key": "value"}'
              error={errors.config?.parameters?.message}
              rows={6}
              monospace
            />
          )}
        />
        <div className="flex justify-between items-center">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setShowTemplates(!showTemplates)}
          >
            {showTemplates ? "Ocultar" : "Mostrar"} Plantillas
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onFormatJSON}
          >
            Formatear JSON
          </Button>
        </div>
        {showTemplates && (
          <JSONTemplates
            templates={JSON_TEMPLATES}
            onSelect={(template) => {
              try {
                const parsed = JSON.parse(template.value);
                setValue("config.parameters", parsed, { shouldValidate: true });
              } catch {
                // Invalid template
              }
            }}
          />
        )}
      </div>
    </FormField>
  );
};

