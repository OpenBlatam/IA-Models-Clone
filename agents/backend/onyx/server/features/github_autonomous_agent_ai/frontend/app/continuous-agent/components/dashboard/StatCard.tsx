import React from "react";
import { formatNumber } from "../../utils/formatting";
import { cn } from "../../utils/classNames";

type StatCardProps = {
  readonly label: string;
  readonly value: number | string;
  readonly subtitle?: string;
  readonly ariaLabel?: string;
  readonly valueClassName?: string;
};

export const StatCard = ({
  label,
  value,
  subtitle,
  ariaLabel,
  valueClassName,
}: StatCardProps): JSX.Element => {
  const displayValue = typeof value === "number" ? formatNumber(value) : value;

  return (
    <div
      className="bg-muted rounded-lg p-4"
      role="listitem"
      aria-label={ariaLabel || `${label}: ${displayValue}`}
    >
      <div className="text-sm text-muted-foreground mb-1">{label}</div>
      <div
        className={cn("text-2xl font-bold", valueClassName)}
        aria-live="polite"
      >
        {displayValue}
      </div>
      {subtitle && (
        <div className="text-xs text-muted-foreground mt-1">{subtitle}</div>
      )}
    </div>
  );
};





