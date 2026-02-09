"use client";

import React from "react";
import { cn } from "../../utils/classNames";

type QuickButtonProps = {
  readonly label: string;
  readonly onClick: () => void;
  readonly title?: string;
  readonly ariaLabel?: string;
  readonly variant?: "default" | "primary" | "secondary";
  readonly size?: "sm" | "md";
  readonly disabled?: boolean;
  readonly className?: string;
};

const VARIANT_CLASSES = {
  default: "bg-gray-100 hover:bg-gray-200 text-gray-700",
  primary: "bg-blue-100 hover:bg-blue-200 text-blue-700",
  secondary: "bg-gray-50 hover:bg-gray-100 text-gray-600 border border-gray-300",
} as const;

const SIZE_CLASSES = {
  sm: "px-2 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
} as const;

export const QuickButton = ({
  label,
  onClick,
  title,
  ariaLabel,
  variant = "default",
  size = "sm",
  disabled = false,
  className,
}: QuickButtonProps): JSX.Element => {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel || title || label}
      className={cn(
        "rounded transition-colors font-medium",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className
      )}
    >
      {label}
    </button>
  );
};







