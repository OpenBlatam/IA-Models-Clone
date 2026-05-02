'use client';

import { motion } from 'framer-motion';
import { cn } from '@/src/utils';

type SpinnerSize = 'sm' | 'md' | 'lg' | 'xl';

interface LoadingSpinnerProps {
    size?: SpinnerSize;
    className?: string;
    label?: string;
}

const sizeClasses: Record<SpinnerSize, { spinner: string; text: string }> = {
    sm: { spinner: 'w-4 h-4 border-2', text: 'text-xs' },
    md: { spinner: 'w-8 h-8 border-3', text: 'text-sm' },
    lg: { spinner: 'w-12 h-12 border-4', text: 'text-base' },
    xl: { spinner: 'w-16 h-16 border-4', text: 'text-lg' },
};

export const LoadingSpinner = ({
    size = 'md',
    className = '',
    label = 'Loading...',
}: LoadingSpinnerProps) => {
    const { spinner, text } = sizeClasses[size];

    return (
        <div
            className={cn('flex flex-col items-center justify-center gap-3', className)}
            role="status"
            aria-label={label}
        >
            <motion.div
                className={cn(
                    spinner,
                    'border-primary/20 border-t-primary rounded-full'
                )}
                animate={{ rotate: 360 }}
                transition={{
                    duration: 1,
                    repeat: Infinity,
                    ease: 'linear',
                }}
                aria-hidden="true"
            />
            <motion.span
                className={cn(text, 'text-text-muted font-medium')}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
            >
                {label}
            </motion.span>
        </div>
    );
};

export const LoadingDots = ({ className = '' }: { className?: string }) => {
    return (
        <div
            className={cn('flex items-center gap-1', className)}
            role="status"
            aria-label="Loading"
        >
            {[0, 1, 2].map((i) => (
                <motion.div
                    key={i}
                    className="w-2 h-2 bg-primary rounded-full"
                    animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                        duration: 1,
                        repeat: Infinity,
                        delay: i * 0.2,
                    }}
                    aria-hidden="true"
                />
            ))}
        </div>
    );
};

export const LoadingPulse = ({
    className = '',
    label = 'Processing...',
}: {
    className?: string;
    label?: string;
}) => {
    return (
        <div
            className={cn('flex flex-col items-center justify-center gap-4', className)}
            role="status"
            aria-label={label}
        >
            <motion.div
                className="w-20 h-20 rounded-full bg-gradient-to-r from-primary to-secondary"
                animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.7, 0.3, 0.7],
                }}
                transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: 'easeInOut',
                }}
                aria-hidden="true"
            />
            <span className="text-text-muted font-medium">{label}</span>
        </div>
    );
};

export const LoadingOverlay = ({
    label = 'Loading...',
}: {
    label?: string;
}) => {
    return (
        <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role="dialog"
            aria-modal="true"
            aria-label={label}
        >
            <LoadingPulse label={label} />
        </motion.div>
    );
};

export default LoadingSpinner;
