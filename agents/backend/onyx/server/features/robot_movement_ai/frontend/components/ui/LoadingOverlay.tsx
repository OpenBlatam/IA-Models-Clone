'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils/cn';
import LoadingSpinner from './LoadingSpinner';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  fullScreen?: boolean;
  className?: string;
  variant?: 'default' | 'dots' | 'pulse' | 'ring';
}

export default function LoadingOverlay({
  isLoading,
  message,
  fullScreen = false,
  className,
  variant = 'default',
}: LoadingOverlayProps) {
  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className={cn(
            'fixed inset-0 z-50 flex items-center justify-center',
            fullScreen ? 'bg-white/95 backdrop-blur-sm' : 'bg-black/20 backdrop-blur-sm',
            className
          )}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="flex flex-col items-center gap-4"
          >
            <LoadingSpinner size="lg" variant={variant} color={fullScreen ? 'blue' : 'white'} />
            {message && (
              <p className={cn('text-sm font-medium', fullScreen ? 'text-tesla-black' : 'text-white')}>
                {message}
              </p>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}



