'use client';

import { forwardRef, type InputHTMLAttributes } from 'react';
import { cn } from '@/src/lib/utils';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    hint?: string;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
    (
        {
            className,
            label,
            error,
            hint,
            leftIcon,
            rightIcon,
            id,
            disabled,
            ...props
        },
        ref
    ) => {
        const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
        const hasError = !!error;

        return (
            <div className={cn("w-full", className)}>
                {label && (
                    <label
                        htmlFor={inputId}
                        className="block text-sm font-medium text-text mb-2"
                    >
                        {label}
                    </label>
                )}
                <div className="relative">
                    {leftIcon && (
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" aria-hidden="true">
                            {leftIcon}
                        </div>
                    )}
                    <input
                        ref={ref}
                        id={inputId}
                        disabled={disabled}
                        className={cn(
                            "w-full px-4 py-3 rounded-xl",
                            "bg-card border text-text",
                            "placeholder:text-text-muted/50",
                            "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background",
                            "transition-all duration-200",
                            "disabled:opacity-50 disabled:cursor-not-allowed",
                            leftIcon && "pl-10",
                            rightIcon && "pr-10",
                            hasError 
                                ? "border-accent-error/50 focus:ring-accent-error focus:border-accent-error"
                                : "border-white/10 focus:ring-primary focus:border-primary/50"
                        )}
                        aria-invalid={hasError}
                        aria-describedby={hasError ? `${inputId}-error` : hint ? `${inputId}-hint` : undefined}
                        {...props}
                    />
                    {rightIcon && (
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted" aria-hidden="true">
                            {rightIcon}
                        </div>
                    )}
                </div>
                {error && (
                    <p id={`${inputId}-error`} className="mt-2 text-sm text-accent-error" role="alert">
                        {error}
                    </p>
                )}
                {hint && !error && (
                    <p id={`${inputId}-hint`} className="mt-2 text-sm text-text-muted">
                        {hint}
                    </p>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';

export default Input;
