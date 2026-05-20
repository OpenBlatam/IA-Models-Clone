import React from "react";
import { cn } from "../../utils/classNames";
import { PromptValidationBadge } from "../ui/PromptValidationBadge";
import { PromptSyntaxHighlighter } from "../ui/PromptSyntaxHighlighter";
import { getPromptSummary, isPerplexityStylePrompt } from "../../utils/prompt-validation";
import { truncateText, shouldTruncate, formatCharacterCount } from "../../utils/text-utils";
import { useCopyToClipboard } from "../../hooks/useCopyToClipboard";
import { useExpandable } from "../../hooks/useExpandable";

type AgentGoalProps = {
  readonly goal?: string;
  readonly className?: string;
};

/**
 * Component to display agent goal/prompt in a collapsible format
 * 
 * Features:
 * - Collapsible display for long prompts
 * - Syntax highlighting preview
 * - Copy to clipboard functionality
 * - Character count display
 * 
 * @param props - Component props
 * @returns The rendered goal component
 */
export const AgentGoal = ({ goal, className }: AgentGoalProps): JSX.Element | null => {
  const { isCopied, copy } = useCopyToClipboard({
    successMessage: "Objetivo copiado al portapapeles",
    errorMessage: "Error al copiar objetivo",
  });
  const { isExpanded, toggle } = useExpandable();

  if (!goal?.trim()) {
    return null;
  }

  const TRUNCATE_LENGTH = 200;
  const truncatedGoal = truncateText(goal, TRUNCATE_LENGTH);
  const needsTruncation = shouldTruncate(goal, TRUNCATE_LENGTH);

  const handleCopy = (): void => {
    copy(goal);
  };

  return (
    <div className={cn("border rounded-lg p-4 bg-gray-50", className)}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <h4 className="text-sm font-semibold text-gray-700">Objetivo/Prompt</h4>
          <span className="text-xs text-gray-500">
            ({formatCharacterCount(goal.length)} caracteres)
          </span>
          <PromptValidationBadge prompt={goal} showDetails={false} />
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleCopy}
            className="text-xs px-2 py-1 text-gray-600 hover:text-gray-900 hover:bg-gray-200 rounded transition-colors"
            aria-label="Copiar objetivo"
            title="Copiar objetivo al portapapeles"
          >
            {isCopied ? "✓ Copiado" : "Copiar"}
          </button>
          {needsTruncation && (
            <button
              type="button"
              onClick={toggle}
              className="text-xs px-2 py-1 text-gray-600 hover:text-gray-900 hover:bg-gray-200 rounded transition-colors"
              aria-label={isExpanded ? "Contraer" : "Expandir"}
            >
              {isExpanded ? "Contraer" : "Expandir"}
            </button>
          )}
        </div>
      </div>
      <div className="relative">
        {isPerplexityStylePrompt(goal) ? (
          <div className={cn("overflow-hidden", !isExpanded && needsTruncation && "max-h-48")}>
            <PromptSyntaxHighlighter
              prompt={isExpanded || !needsTruncation ? goal : truncatedGoal}
              maxHeight={isExpanded ? "600px" : "192px"}
            />
            {!isExpanded && needsTruncation && (
              <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-gray-50 to-transparent pointer-events-none" />
            )}
          </div>
        ) : (
          <pre
            className={cn(
              "text-xs font-mono text-gray-800 whitespace-pre-wrap break-words overflow-hidden bg-white p-3 rounded border",
              !isExpanded && needsTruncation && "line-clamp-6"
            )}
          >
            {isExpanded || !needsTruncation ? goal : truncatedGoal}
          </pre>
        )}
        {!isExpanded && needsTruncation && !isPerplexityStylePrompt(goal) && (
          <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-gray-50 to-transparent pointer-events-none" />
        )}
      </div>
      <div className="mt-2 pt-2 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          {getPromptSummary(goal)}
        </div>
      </div>
    </div>
  );
};

