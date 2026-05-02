'use client';

import { forwardRef } from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { cn } from '@/src/utils';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'success';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'className' | 'children'> {
    className?: string;
    variant?: ButtonVariant;
    size?: ButtonSize;
    isLoading?: boolean;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
    children?: React.ReactNode;
}

const variantClasses: Record<ButtonVariant, string> = {
    primary: `
    bg-gradient-to-r from-primary to-primary-glow 
    hover:from-primary-glow hover:to-primary
    text-white shadow-lg shadow-primary/25
    hover:shadow-xl hover:shadow-primary/40
  `,
    secondary: `
    bg-card hover:bg-card-hover
    text-text border border-primary/30
    hover:border-primary/60
  `,
    ghost: `
    bg-transparent hover:bg-card
    text-text-muted hover:text-text
  `,
    danger: `
    bg-gradient-to-r from-accent-error to-red-600
    hover:from-red-600 hover:to-accent-error
    text-white shadow-lg shadow-accent-error/25
  `,
    success: `
    bg-gradient-to-r from-accent-success to-emerald-500
    hover:from-emerald-500 hover:to-accent-success
    text-white shadow-lg shadow-accent-success/25
  `,
};

const sizeClasses: Record<ButtonSize, string> = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-5 py-2.5 text-base',
    lg: 'px-7 py-3.5 text-lg',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    (
        {
            className,
            variant = 'primary',
            size = 'md',
            isLoading = false,
            disabled,
            leftIcon,
            rightIcon,
            children,
            ...props
        },
        ref
    ) => {
        const isDisabled = disabled || isLoading;

        return (
            <motion.button
                ref={ref}
                whileHover={{ scale: isDisabled ? 1 : 1.02 }}
                whileTap={{ scale: isDisabled ? 1 : 0.98 }}
                className={cn(
                    'inline-flex items-center justify-center gap-2',
                    'font-semibold rounded-xl',
                    'transition-all duration-300 ease-out',
                    'focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:ring-offset-background',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    variantClasses[variant],
                    sizeClasses[size],
                    className
                )}
                disabled={isDisabled}
                aria-disabled={isDisabled}
                tabIndex={isDisabled ? -1 : 0}
                {...props}
            >
                {isLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" aria-hidden="true" />
                ) : leftIcon ? (
                    <span className="flex-shrink-0" aria-hidden="true">{leftIcon}</span>
                ) : null}
                {children}
                {!isLoading && rightIcon && (
                    <span className="flex-shrink-0" aria-hidden="true">{rightIcon}</span>
                )}
            </motion.button>
        );
    }
);

Button.displayName = 'Button';

export default Button;
