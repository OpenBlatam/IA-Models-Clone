'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface CTASectionProps {
  title: string;
  description?: string;
  primaryAction?: {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary';
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  children?: ReactNode;
  className?: string;
  background?: 'white' | 'gray' | 'gradient';
}

export default function CTASection({
  title,
  description,
  primaryAction,
  secondaryAction,
  children,
  className,
  background = 'white',
}: CTASectionProps) {
  const backgroundClasses = {
    white: 'bg-white',
    gray: 'bg-gray-50',
    gradient: 'bg-gradient-to-br from-gray-50 to-white',
  };

  return (
    <section className={cn('py-16 md:py-24', backgroundClasses[background], className)}>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="space-y-6"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-tesla-black tracking-tight">
            {title}
          </h2>
          {description && (
            <p className="text-lg md:text-xl text-tesla-gray-dark max-w-2xl mx-auto">
              {description}
            </p>
          )}
          {(primaryAction || secondaryAction) && (
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-8">
              {primaryAction && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={primaryAction.onClick}
                  className={cn(
                    'px-8 py-4 rounded-md font-medium text-base transition-all min-h-[52px]',
                    primaryAction.variant === 'secondary'
                      ? 'bg-white border-2 border-tesla-black text-tesla-black hover:bg-tesla-black hover:text-white'
                      : 'bg-tesla-blue text-white hover:bg-opacity-90'
                  )}
                >
                  {primaryAction.label}
                </motion.button>
              )}
              {secondaryAction && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={secondaryAction.onClick}
                  className="px-8 py-4 rounded-md font-medium text-base bg-transparent border-2 border-gray-300 text-tesla-black hover:border-gray-400 transition-all min-h-[52px]"
                >
                  {secondaryAction.label}
                </motion.button>
              )}
            </div>
          )}
          {children}
        </motion.div>
      </div>
    </section>
  );
}



