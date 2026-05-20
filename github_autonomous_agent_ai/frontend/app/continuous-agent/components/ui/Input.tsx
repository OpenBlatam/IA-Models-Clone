"use client";

import { forwardRef } from "react";
import type { ChangeEvent } from "react";
import { cn } from "../../utils/classNames";

type InputProps = {
  readonly id: string;
  readonly type?: "text" | "number" | "email" | "password";
  readonly value: string | number;
  readonly onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  readonly placeholder?: string;
  readonly required?: boolean;
  readonly disabled?: boolean;
  readonly min?: number;
  readonly max?: number;
  readonly maxLength?: number;
  readonly step?: number;
  readonly error?: string | null;
  readonly className?: string;
  readonly ariaLabel?: string;
  readonly ariaDescribedBy?: string;
};

export const Input = forwardRef<HTMLInputElement, Omit<InputProps, "ref">>(
  (
    {
      id,
      type = "text",
      value,
      onChange,
      placeholder,
      required = false,
      disabled = false,
      min,
      max,
      maxLength,
      step,
      error,
      className,
      ariaLabel,
      ariaDescribedBy,
    },
    ref
  ): JSX.Element => {
    return (
      <input
        ref={ref}
        id={id}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        min={min}
        max={max}
        maxLength={maxLength}
        step={step}
        aria-label={ariaLabel}
        aria-required={required}
        aria-invalid={!!error}
        aria-describedby={ariaDescribedBy}
        className={cn(
          "w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors",
          error && "border-red-300 focus:ring-red-500",
          disabled && "opacity-50 cursor-not-allowed bg-gray-50",
          className
        )}
      />
    );
  }
);

Input.displayName = "Input";

