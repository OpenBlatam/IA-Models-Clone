import React, { useState, useCallback } from "react";
import { cn } from "../../utils/classNames";
import type { PromptTemplate } from "../../constants/prompt-templates";

type PromptTemplatePreviewProps = {
  readonly template: PromptTemplate;
  readonly onApply: (template: PromptTemplate) => void;
  readonly onClose?: () => void;
};

/**
 * Component to preview a prompt template before applying it
 * 
 * Features:
 * - Preview of template content
 * - Character count
 * - Apply or cancel actions
 * - Scrollable content area
 * 
 * @param props - Component props
 * @returns The rendered preview component
 */
export const PromptTemplatePreview = ({
  template,
  onApply,
  onClose,
}: PromptTemplatePreviewProps): JSX.Element => {
  const [isApplying, setIsApplying] = useState(false);

  const characterCount = template.value.length;
  const lineCount = template.value.split("\n").length;

  const handleApply = useCallback(async (): Promise<void> => {
    setIsApplying(true);
    try {
      onApply(template);
    } finally {
      setIsApplying(false);
    }
  }, [template, onApply]);

  return (
    <div className="border rounded-lg bg-white shadow-lg max-w-4xl max-h-[80vh] flex flex-col">
      {/* Header */}
      <div className="p-4 border-b bg-gray-50 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{template.name}</h3>
            <p className="text-sm text-gray-600 mt-1">{template.description}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              {characterCount.toLocaleString()} caracteres • {lineCount} líneas
            </span>
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors"
                aria-label="Cerrar preview"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        <pre className="text-xs font-mono text-gray-800 whitespace-pre-wrap break-words">
          {template.value}
        </pre>
      </div>

      {/* Footer */}
      <div className="p-4 border-t bg-gray-50 rounded-b-lg flex items-center justify-between">
        <div className="text-xs text-gray-500">
          Categoría:{" "}
          <span className="font-medium">
            {template.category === "research" && "Investigación"}
            {template.category === "content" && "Contenido"}
            {template.category === "analysis" && "Análisis"}
            {template.category === "custom" && "Personalizado"}
          </span>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
            >
              Cancelar
            </button>
          )}
          <button
            type="button"
            onClick={handleApply}
            disabled={isApplying}
            className={cn(
              "px-4 py-2 text-sm font-medium text-white rounded-md transition-colors",
              isApplying
                ? "bg-blue-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            )}
          >
            {isApplying ? "Aplicando..." : "Aplicar Plantilla"}
          </button>
        </div>
      </div>
    </div>
  );
};




