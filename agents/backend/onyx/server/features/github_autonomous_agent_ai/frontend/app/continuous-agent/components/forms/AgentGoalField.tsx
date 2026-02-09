import React, { useRef, useState, useMemo, useCallback, memo } from "react";
import { FormField } from "../ui/FormField";
import { Textarea } from "../ui/Textarea";
import { QuickButton } from "../ui/QuickButton";
import { PromptTemplatePreview } from "../ui/PromptTemplatePreview";
import { PromptValidationBadge } from "../ui/PromptValidationBadge";
import { validateGoal } from "../../utils/validation";
import { PROMPT_TEMPLATES, type PromptTemplate } from "../../constants/prompt-templates";
import { exportPrompt, importPrompt, copyPromptToClipboard } from "../../utils/prompt-export";
import { cn } from "../../utils/classNames";
import type { UseAgentFormReturn } from "../../hooks/useAgentForm";
import { useDebouncedCallback } from "use-debounce";

const VALIDATION_DEBOUNCE_MS = 300;

type AgentGoalFieldProps = {
  readonly form: UseAgentFormReturn;
  readonly inputRef?: React.RefObject<HTMLTextAreaElement>;
  readonly onErrorChange?: (error: string | null) => void;
  readonly onValidate?: () => void;
};

/**
 * Form field component for agent goal/prompt
 * 
 * Features:
 * - Goal validation with real-time feedback
 * - Prompt template support
 * - Character count indicator
 * - Syntax highlighting support
 * 
 * @param props - Component props
 * @returns The rendered goal field component
 */
const AgentGoalFieldComponent = ({
  form,
  inputRef,
  onErrorChange,
  onValidate,
}: AgentGoalFieldProps): JSX.Element => {
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const ref = inputRef || internalRef;
  const [showTemplates, setShowTemplates] = useState(false);
  const [previewTemplate, setPreviewTemplate] = useState<PromptTemplate | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);

  const debouncedValidate = useDebouncedCallback(() => {
    onValidate?.();
  }, VALIDATION_DEBOUNCE_MS);

  const goalValidation = useMemo(() => {
    return validateGoal(form.goal);
  }, [form.goal]);

  const isGoalValid = useMemo((): boolean => {
    return goalValidation.isValid;
  }, [goalValidation]);

  const goalErrorMessage = useMemo((): string | undefined => {
    if (!form.goal?.trim()) {
      return undefined; // Optional field
    }
    if (!isGoalValid) {
      return goalValidation.error || "Objetivo inválido";
    }
    return undefined;
  }, [form.goal, isGoalValid, goalValidation]);

  const characterCount = useMemo((): number => {
    return form.goal?.length || 0;
  }, [form.goal]);

  const maxCharacters = 10000;

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>): void => {
      const value = event.target.value;
      form.setGoal(value);
      onErrorChange?.(null);

      if (value.trim()) {
        if (form.errors.goal) {
          debouncedValidate();
        }
      } else {
        debouncedValidate();
      }
    },
    [form, debouncedValidate, onErrorChange]
  );

  const applyTemplate = useCallback(
    (template: PromptTemplate): void => {
      form.setGoal(template.value);
      setShowTemplates(false);
      setPreviewTemplate(null);
      setSearchQuery("");
      onErrorChange?.(null);
    },
    [form, onErrorChange]
  );

  const handlePreviewTemplate = useCallback(
    (template: PromptTemplate): void => {
      setPreviewTemplate(template);
    },
    []
  );

  const handleClosePreview = useCallback((): void => {
    setPreviewTemplate(null);
  }, []);

  const clearGoal = useCallback((): void => {
    form.setGoal("");
    setShowTemplates(false);
    onErrorChange?.(null);
  }, [form, onErrorChange]);

  const handleExport = useCallback(async (): Promise<void> => {
    if (!form.goal?.trim()) {
      onErrorChange?.("No hay prompt para exportar");
      return;
    }

    setIsExporting(true);
    try {
      const agentName = form.name.trim() || "agent";
      const sanitizedName = agentName.replace(/[^a-z0-9]/gi, "_").toLowerCase();
      exportPrompt(form.goal, `${sanitizedName}_prompt.txt`);
    } catch (error) {
      onErrorChange?.(error instanceof Error ? error.message : "Error al exportar");
    } finally {
      setIsExporting(false);
    }
  }, [form.goal, form.name, onErrorChange]);

  const handleImport = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }

      setIsImporting(true);
      try {
        const importedPrompt = await importPrompt(file);
        form.setGoal(importedPrompt);
        onErrorChange?.(null);
      } catch (error) {
        onErrorChange?.(error instanceof Error ? error.message : "Error al importar");
      } finally {
        setIsImporting(false);
        // Reset input to allow selecting same file again
        event.target.value = "";
      }
    },
    [form, onErrorChange]
  );

  const handleCopy = useCallback(async (): Promise<void> => {
    if (!form.goal?.trim()) {
      return;
    }

    try {
      await copyPromptToClipboard(form.goal);
      onErrorChange?.(null);
    } catch (error) {
      onErrorChange?.(error instanceof Error ? error.message : "Error al copiar");
    }
  }, [form.goal, onErrorChange]);

  const filteredTemplates = useMemo(() => {
    if (!searchQuery.trim()) {
      return PROMPT_TEMPLATES;
    }
    const query = searchQuery.toLowerCase();
    return PROMPT_TEMPLATES.filter(
      (template) =>
        template.name.toLowerCase().includes(query) ||
        template.description.toLowerCase().includes(query) ||
        template.category.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  const templatesByCategory = useMemo(() => {
    const categories: Record<string, PromptTemplate[]> = {
      research: [],
      content: [],
      analysis: [],
      technical: [],
      support: [],
      custom: [],
    };

    filteredTemplates.forEach((template) => {
      if (categories[template.category]) {
        categories[template.category].push(template);
      }
    });

    return categories;
  }, [filteredTemplates]);

  return (
    <FormField
      label="Objetivo/Prompt del Agente (Opcional)"
      htmlFor="goal"
      error={form.errors.goal || goalErrorMessage}
      helpText="Define el objetivo o prompt del agente. Puedes usar plantillas predefinidas o escribir uno personalizado."
    >
      <div className="space-y-2">
        <div className="flex gap-2 items-start">
          <div className="relative flex-1">
            <Textarea
              id="goal"
              ref={ref}
              value={form.goal || ""}
              onChange={handleChange}
              rows={12}
              placeholder="Escribe el objetivo o prompt del agente aquí... (opcional)"
              error={form.errors.goal || goalErrorMessage}
              monospace
              ariaLabel="Objetivo o prompt del agente"
              ariaDescribedBy={
                form.errors.goal || (!isGoalValid && form.goal?.trim())
                  ? "goal-error"
                  : "goal-help"
              }
            />
            {form.goal && (
              <div
                className={cn(
                  "absolute bottom-2 right-2 text-xs font-semibold px-2 py-1 rounded pointer-events-none transition-all duration-200",
                  characterCount > maxCharacters
                    ? "bg-red-100 text-red-700 shadow-sm"
                    : characterCount > maxCharacters * 0.9
                    ? "bg-yellow-100 text-yellow-700 shadow-sm"
                    : "bg-gray-100 text-gray-600 shadow-sm"
                )}
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                {characterCount.toLocaleString()} / {maxCharacters.toLocaleString()}
              </div>
            )}
          </div>
          <div className="flex flex-col gap-1">
            <QuickButton
              label="Plantillas"
              onClick={() => setShowTemplates(!showTemplates)}
              title="Mostrar plantillas de prompts"
              ariaLabel="Mostrar plantillas de prompts"
              variant="default"
              size="md"
            />
            {form.goal && (
              <>
                <QuickButton
                  label="Copiar"
                  onClick={handleCopy}
                  title="Copiar prompt al portapapeles"
                  ariaLabel="Copiar prompt"
                  variant="default"
                  size="md"
                />
                <QuickButton
                  label="Exportar"
                  onClick={handleExport}
                  title="Exportar prompt a archivo"
                  ariaLabel="Exportar prompt"
                  variant="default"
                  size="md"
                  disabled={isExporting}
                />
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".txt,.md"
                    onChange={handleImport}
                    className="hidden"
                    disabled={isImporting}
                  />
                  <span
                    className="inline-flex items-center justify-center px-3 py-1.5 text-sm rounded transition-colors font-medium bg-gray-50 hover:bg-gray-100 text-gray-600 border border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
                    title="Importar prompt desde archivo"
                    aria-label="Importar prompt"
                  >
                    {isImporting ? "Importando..." : "Importar"}
                  </span>
                </label>
                <QuickButton
                  label="Limpiar"
                  onClick={clearGoal}
                  title="Limpiar objetivo"
                  ariaLabel="Limpiar objetivo"
                  variant="secondary"
                  size="md"
                />
              </>
            )}
          </div>
        </div>
        {showTemplates && (
          <div className="border rounded-lg p-4 bg-gray-50 space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-gray-700">
                Plantillas de Prompts
              </div>
              <button
                type="button"
                onClick={() => setShowTemplates(false)}
                className="text-xs text-gray-500 hover:text-gray-700"
              >
                Cerrar
              </button>
            </div>
            
            {/* Search */}
            <div>
              <input
                type="text"
                placeholder="Buscar plantillas..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {filteredTemplates.length === 0 ? (
              <div className="text-center py-8 text-sm text-gray-500">
                No se encontraron plantillas que coincidan con "{searchQuery}"
              </div>
            ) : (
              Object.entries(templatesByCategory).map(([category, templates]) => {
                if (templates.length === 0) return null;
                return (
                  <div key={category} className="space-y-2">
                    <div className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                      {category === "research" && "Investigación"}
                      {category === "content" && "Contenido"}
                      {category === "analysis" && "Análisis"}
                      {category === "technical" && "Técnico"}
                      {category === "support" && "Soporte"}
                      {category === "custom" && "Personalizado"}
                    </div>
                    <div className="grid grid-cols-1 gap-2">
                      {templates.map((template) => (
                        <div
                          key={template.name}
                          className="flex items-center gap-2 p-3 bg-white border rounded hover:border-blue-500 hover:bg-blue-50 transition-colors"
                        >
                          <div className="flex-1 text-left">
                            <div className="font-medium text-sm text-gray-900">
                              {template.name}
                            </div>
                            <div className="text-xs text-gray-600 mt-1">
                              {template.description}
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              {template.value.length.toLocaleString()} caracteres
                            </div>
                          </div>
                          <div className="flex gap-1">
                            <button
                              type="button"
                              onClick={() => handlePreviewTemplate(template)}
                              className="px-2 py-1 text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-100 rounded transition-colors"
                              aria-label={`Vista previa de ${template.name}`}
                            >
                              Vista previa
                            </button>
                            <button
                              type="button"
                              onClick={() => applyTemplate(template)}
                              className="px-2 py-1 text-xs text-white bg-blue-600 hover:bg-blue-700 rounded transition-colors"
                              aria-label={`Aplicar plantilla ${template.name}`}
                            >
                              Aplicar
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {previewTemplate && (
          <div className="mt-4">
            <PromptTemplatePreview
              template={previewTemplate}
              onApply={applyTemplate}
              onClose={handleClosePreview}
            />
          </div>
        )}

        {form.goal && form.goal.trim() && (
          <div className="mt-2">
            <PromptValidationBadge prompt={form.goal} showDetails={false} />
          </div>
        )}
      </div>
    </FormField>
  );
};

// Memoize component to prevent unnecessary re-renders
export const AgentGoalField = memo(AgentGoalFieldComponent, (prevProps, nextProps) => {
  return (
    prevProps.form.goal === nextProps.form.goal &&
    prevProps.form.errors.goal === nextProps.form.errors.goal &&
    prevProps.inputRef === nextProps.inputRef &&
    prevProps.onErrorChange === nextProps.onErrorChange &&
    prevProps.onValidate === nextProps.onValidate
  );
});

AgentGoalField.displayName = "AgentGoalField";

