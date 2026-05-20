import React, { useMemo } from "react";
import { cn } from "../../utils/classNames";
import { validatePromptStructure, type PromptValidationResult } from "../../utils/prompt-validation";
import {
  getValidationBadgeStyle,
  getValidationBadgeTitle,
  getValidationBadgeText,
  getValidationBadgeIcon,
} from "../../utils/badge-styles";
import { BADGE_STYLES } from "../../constants/styles";

type PromptValidationBadgeProps = {
  readonly prompt?: string;
  readonly className?: string;
  readonly showDetails?: boolean;
};

/**
 * Component to display prompt validation status
 * 
 * Features:
 * - Visual indicator of prompt structure validity
 * - Shows errors and warnings count
 * - Optional detailed view
 * 
 * @param props - Component props
 * @returns The rendered validation badge component
 */
export const PromptValidationBadge = ({
  prompt,
  className,
  showDetails = false,
}: PromptValidationBadgeProps): JSX.Element | null => {
  const validation = useMemo((): PromptValidationResult | null => {
    if (!prompt?.trim()) {
      return null;
    }
    return validatePromptStructure(prompt);
  }, [prompt]);

  if (!validation) {
    return null;
  }

  const { isValid, errors, warnings, stats } = validation;

  if (!showDetails) {
    // Simple badge
    const badgeStyle = getValidationBadgeStyle(validation);
    const badgeTitle = getValidationBadgeTitle(validation);
    const badgeText = getValidationBadgeText(validation);
    const badgeIcon = getValidationBadgeIcon(validation);

    return (
      <div
        className={cn(BADGE_STYLES.BASE, badgeStyle, className)}
        title={badgeTitle}
      >
        <span>{badgeIcon}</span>
        <span>{badgeText}</span>
      </div>
    );
  }

  // Detailed view
  return (
    <div className={cn("border rounded-lg p-3 bg-gray-50 space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-700">Validación del Prompt</span>
        <div className={cn("px-2 py-1 rounded text-xs font-medium", getValidationBadgeStyle(validation))}>
          {isValid && errors.length === 0 ? "Válido" : errors.length > 0 ? "Con Errores" : "Con Advertencias"}
        </div>
      </div>

      {errors.length > 0 && (
        <div>
          <div className="text-xs font-medium text-red-700 mb-1">Errores:</div>
          <ul className="list-disc list-inside text-xs text-red-600 space-y-1">
            {errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {warnings.length > 0 && (
        <div>
          <div className="text-xs font-medium text-yellow-700 mb-1">Advertencias:</div>
          <ul className="list-disc list-inside text-xs text-yellow-600 space-y-1">
            {warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="pt-2 border-t border-gray-200">
        <div className="text-xs text-gray-600 space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">Estructura:</span>
            <div className="flex gap-1 flex-wrap">
              {stats.hasGoalTag && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">Goal</span>
              )}
              {stats.hasFormatRules && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">Format</span>
              )}
              {stats.hasRestrictions && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">Restrictions</span>
              )}
              {stats.hasQueryType && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">Query Type</span>
              )}
              {stats.hasPlanningRules && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">Planning</span>
              )}
              {stats.hasOutput && (
                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">Output</span>
              )}
            </div>
          </div>
          <div>
            <span className="font-medium">{stats.tagCount}</span> tags •{" "}
            <span className="font-medium">{stats.characterCount.toLocaleString()}</span> caracteres •{" "}
            <span className="font-medium">{stats.lineCount}</span> líneas
          </div>
        </div>
      </div>
    </div>
  );
};


