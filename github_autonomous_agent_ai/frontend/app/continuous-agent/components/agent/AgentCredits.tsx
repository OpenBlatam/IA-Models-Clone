import React from "react";
import {
  formatCredits,
  getCreditsStatusClass,
  getCreditsStatusAriaLabel,
} from "../../utils/formatting";
import { cn } from "../../utils/classNames";

type AgentCreditsProps = {
  readonly creditsRemaining: number | null;
};

export const AgentCredits = ({ creditsRemaining }: AgentCreditsProps): JSX.Element => {
  const creditsClass = getCreditsStatusClass(creditsRemaining);
  const creditsDisplay = formatCredits(creditsRemaining);
  const creditsAriaLabel = getCreditsStatusAriaLabel(creditsRemaining);

  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">Créditos restantes:</span>
      <span className={creditsClass} aria-label={creditsAriaLabel}>
        {creditsDisplay}
      </span>
    </div>
  );
};





