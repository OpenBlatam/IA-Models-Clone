"use client";

import React from "react";
import { cn } from "../../utils/classNames";

type TextareaProps = {
  readonly id: string;
  readonly value: string;
  readonly onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  readonly placeholder?: string;
  readonly required?: boolean;
  readonly disabled?: boolean;
  readonly rows?: number;
  readonly maxLength?: number;
  readonly error?: string | null;
  readonly className?: string;
  readonly ariaLabel?: string;
  readonly ariaDescribedBy?: string;
  readonly monospace?: boolean;
  readonly ref?: React.Ref<HTMLTextAreaElement>;
};

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  Omit<TextareaProps, "ref">
>(
  (
    {
      id,
      value,
      onChange,
      placeholder,
      required = false,
      disabled = false,
      rows = 3,
      maxLength,
      error,
      className,
      ariaLabel,
      ariaDescribedBy,
      monospace = false,
    },
    ref
  ): JSX.Element => {
    return (
      <textarea
        ref={ref}
        id={id}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        rows={rows}
        maxLength={maxLength}
        aria-label={ariaLabel}
        aria-required={required}
        aria-invalid={!!error}
        aria-describedby={ariaDescribedBy}
        className={cn(
          "w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors resize-y",
          error && "border-red-300 focus:ring-red-500",
          disabled && "opacity-50 cursor-not-allowed bg-gray-50",
          monospace && "font-mono text-sm",
          className
        )}
      />
    );
  }
);

Textarea.displayName = "Textarea";

