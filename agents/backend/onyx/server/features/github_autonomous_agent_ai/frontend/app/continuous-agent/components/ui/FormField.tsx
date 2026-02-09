"use client";

import type { ReactNode } from "react";
import { cn } from "../../utils/classNames";

type FormFieldProps = {
  readonly label: string;
  readonly htmlFor: string;
  readonly required?: boolean;
  readonly error?: string | null;
  readonly helpText?: string;
  readonly children: ReactNode;
  readonly className?: string;
};

export const FormField = ({
  label,
  htmlFor,
  required = false,
  error,
  helpText,
  children,
  className,
}: FormFieldProps): JSX.Element => {
  return (
    <div className={cn("space-y-2", className)}>
      <label htmlFor={htmlFor} className="block text-sm font-medium">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      {children}
      {error && (
        <p className="text-sm text-red-600" role="alert" aria-live="polite">
          {error}
        </p>
      )}
      {helpText && !error && (
        <p className="text-xs text-muted-foreground">{helpText}</p>
      )}
    </div>
  );
};



