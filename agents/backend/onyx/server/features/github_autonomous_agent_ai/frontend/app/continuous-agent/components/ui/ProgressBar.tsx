"use client";

import React from "react";
import { cn } from "../../utils/classNames";

type ProgressBarProps = {
  readonly value: number;
  readonly max?: number;
  readonly showLabel?: boolean;
  readonly size?: "sm" | "md" | "lg";
  readonly variant?: "default" | "success" | "warning" | "error";
  readonly className?: string;
  readonly ariaLabel?: string;
};

export const ProgressBar = ({
  value,
  max = 100,
  showLabel = false,
  size = "md",
  variant = "default",
  className,
  ariaLabel,
}: ProgressBarProps): JSX.Element => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const sizeClasses = {
    sm: "h-1",
    md: "h-2",
    lg: "h-3",
  };

  const variantClasses = {
    default: "bg-blue-600",
    success: "bg-green-600",
    warning: "bg-yellow-600",
    error: "bg-red-600",
  };

  return (
    <div className={cn("w-full", className)}>
      {showLabel && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs text-gray-600">{Math.round(percentage)}%</span>
          <span className="text-xs text-gray-500">
            {value} / {max}
          </span>
        </div>
      )}
      <div
        className={cn(
          "w-full bg-gray-200 rounded-full overflow-hidden",
          sizeClasses[size]
        )}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={ariaLabel || `Progress: ${Math.round(percentage)}%`}
      >
        <div
          className={cn(
            "h-full transition-all duration-300 ease-out rounded-full",
            variantClasses[variant]
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};







