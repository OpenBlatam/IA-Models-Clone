import React, { useRef, useState, useMemo, useCallback, memo } from "react";
import { FormField } from "../ui/FormField";
import { Textarea } from "../ui/Textarea";
import { QuickButton } from "../ui/QuickButton";
import { JSONTemplates } from "../ui/JSONTemplates";
import { validateJSON } from "../../utils/validation";
import { formatJSONError } from "../../utils/formatters";
import { cn } from "../../utils/classNames";
import type { UseAgentFormReturn } from "../../hooks/useAgentForm";
import { useDebouncedCallback } from "use-debounce";

const VALIDATION_DEBOUNCE_MS = 300;

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

type AgentParametersFieldProps = {
  readonly form: UseAgentFormReturn;
  readonly inputRef?: React.RefObject<HTMLTextAreaElement>;
  readonly onErrorChange?: (error: string | null) => void;
  readonly onValidate?: () => void;
};

/**
 * Form field component for agent JSON parameters
 * 
 * Features:
 * - JSON validation with real-time feedback
 * - JSON formatting
 * - Template support
 * - Syntax highlighting indicator
 * 
 * @param props - Component props
 * @returns The rendered parameters field component
 */
const AgentParametersFieldComponent = ({
  form,
  inputRef,
  onErrorChange,
  onValidate,
}: AgentParametersFieldProps): JSX.Element => {
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const ref = inputRef || internalRef;
  const [showJSONTemplates, setShowJSONTemplates] = useState(false);

  const debouncedValidate = useDebouncedCallback(() => {
    onValidate?.();
  }, VALIDATION_DEBOUNCE_MS);

  const jsonValidation = useMemo(() => {
    return validateJSON(form.parameters);
  }, [form.parameters]);

  const isJSONValid = useMemo((): boolean => {
    return jsonValidation.isValid;
  }, [jsonValidation]);

  const jsonErrorMessage = useMemo((): string | undefined => {
    if (!form.parameters.trim()) {
      return undefined;
    }
    if (!isJSONValid) {
      return jsonValidation.error || "JSON inválido";
    }
    return undefined;
  }, [form.parameters, isJSONValid, jsonValidation]);

  const formatJSON = useCallback((): void => {
    if (!form.parameters.trim()) {
      form.setParameters("{}");
      return;
    }

    try {
      const parsed = JSON.parse(form.parameters);
      const formatted = JSON.stringify(parsed, null, 2);
      form.setParameters(formatted);
      onErrorChange?.(null);
    } catch (error) {
      const errorMessage = formatJSONError(error);
      onErrorChange?.(`No se puede formatear JSON inválido: ${errorMessage}`);
    }
  }, [form, onErrorChange]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>): void => {
      const value = event.target.value;
      form.setParameters(value);
      onErrorChange?.(null);

      if (value.trim()) {
        if (form.errors.parameters) {
          debouncedValidate();
        }
      } else {
        debouncedValidate();
      }
    },
    [form, debouncedValidate, onErrorChange]
  );

  const applyJSONTemplate = useCallback(
    (template: { readonly name: string; readonly value: string; readonly description?: string }): void => {
      form.setParameters(template.value);
      setShowJSONTemplates(false);
      onErrorChange?.(null);
    },
    [form, onErrorChange]
  );

  return (
    <FormField
      label="Parámetros (JSON)"
      htmlFor="parameters"
      error={form.errors.parameters || jsonErrorMessage}
      helpText="Configuración adicional en formato JSON. Opcional. Usa Ctrl+K para formatear."
    >
      <div className="space-y-2">
        <div className="flex gap-2 items-start">
          <div className="relative flex-1">
            <Textarea
              id="parameters"
              ref={ref}
              value={form.parameters}
              onChange={handleChange}
              rows={5}
              placeholder='{"key": "value"}'
              error={form.errors.parameters || jsonErrorMessage}
              monospace
              ariaLabel="Parámetros JSON del agente"
              ariaDescribedBy={
                form.errors.parameters || (!isJSONValid && form.parameters.trim())
                  ? "parameters-error"
                  : "parameters-help"
              }
            />
            {form.parameters.trim() && (
              <div
                id="parameters-status"
                className={cn(
                  "absolute top-2 right-2 text-xs font-semibold px-2 py-1 rounded pointer-events-none transition-all duration-200",
                  isJSONValid
                    ? "bg-green-100 text-green-700 shadow-sm"
                    : "bg-red-100 text-red-700 shadow-sm"
                )}
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                {isJSONValid ? (
                  <span className="flex items-center gap-1">
                    <span>✓</span>
                    <span>JSON válido</span>
                  </span>
                ) : (
                  <span className="flex items-center gap-1">
                    <span>✗</span>
                    <span>JSON inválido</span>
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="flex flex-col gap-1">
            <QuickButton
              label="Formatear"
              onClick={formatJSON}
              disabled={!form.parameters.trim() || !isJSONValid}
              title="Formatear JSON (Ctrl+K cuando el campo está enfocado)"
              ariaLabel="Formatear JSON"
              variant="primary"
              size="md"
            />
            <QuickButton
              label="Plantillas"
              onClick={() => setShowJSONTemplates(!showJSONTemplates)}
              title="Plantillas JSON"
              ariaLabel="Mostrar plantillas JSON"
              variant="default"
              size="md"
            />
          </div>
        </div>
        {showJSONTemplates && (
          <JSONTemplates templates={JSON_TEMPLATES} onSelect={applyJSONTemplate} />
        )}
      </div>
    </FormField>
  );
};

// Memoize component to prevent unnecessary re-renders
export const AgentParametersField = memo(AgentParametersFieldComponent, (prevProps, nextProps) => {
  return (
    prevProps.form.parameters === nextProps.form.parameters &&
    prevProps.form.errors.parameters === nextProps.form.errors.parameters &&
    prevProps.inputRef === nextProps.inputRef &&
    prevProps.onErrorChange === nextProps.onErrorChange &&
    prevProps.onValidate === nextProps.onValidate
  );
});

AgentParametersField.displayName = "AgentParametersField";


