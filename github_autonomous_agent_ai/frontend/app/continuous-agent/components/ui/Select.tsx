"use client";

import React from "react";
import { cn } from "../../utils/classNames";

type SelectOption = {
  readonly value: string;
  readonly label: string;
};

type SelectProps = {
  readonly id: string;
  readonly value: string;
  readonly onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
  readonly options: readonly SelectOption[];
  readonly required?: boolean;
  readonly disabled?: boolean;
  readonly error?: string | null;
  readonly className?: string;
  readonly ariaLabel?: string;
  readonly ariaDescribedBy?: string;
};

export const Select = ({
  id,
  value,
  onChange,
  options,
  required = false,
  disabled = false,
  error,
  className,
  ariaLabel,
  ariaDescribedBy,
}: SelectProps): JSX.Element => {
  return (
    <select
      id={id}
      value={value}
      onChange={onChange}
      required={required}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-required={required}
      aria-invalid={!!error}
      aria-describedby={ariaDescribedBy}
      className={cn(
        "w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors bg-background",
        error && "border-red-300 focus:ring-red-500",
        disabled && "opacity-50 cursor-not-allowed bg-gray-50",
        className
      )}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
};







