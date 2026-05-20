import React, { useMemo } from "react";
import { highlightPromptTags, getTagColorClass } from "../../utils/prompt-highlighting";
import { cn } from "../../utils/classNames";

type PromptSyntaxHighlighterProps = {
  readonly prompt: string;
  readonly className?: string;
  readonly maxHeight?: string;
};

/**
 * Component to display prompt with basic syntax highlighting
 * 
 * Features:
 * - Highlights XML-like tags
 * - Color-coded tags by type
 * - Monospace font for readability
 * - Scrollable if content is long
 * 
 * @param props - Component props
 * @returns The rendered highlighted prompt component
 */
export const PromptSyntaxHighlighter = ({
  prompt,
  className,
  maxHeight = "400px",
}: PromptSyntaxHighlighterProps): JSX.Element => {
  const highlighted = useMemo(() => highlightPromptTags(prompt), [prompt]);

  return (
    <div
      className={cn(
        "overflow-auto rounded border bg-gray-900 p-4",
        className
      )}
      style={{ maxHeight }}
    >
      <pre
        className="text-xs font-mono text-gray-100 whitespace-pre-wrap break-words"
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />
    </div>
  );
};




