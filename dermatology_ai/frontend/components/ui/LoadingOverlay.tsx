'use client';

import React, { memo } from 'react';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  subMessage?: string;
  className?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = memo(({
  isLoading,
  message = 'Loading...',
  subMessage,
  className,
}) => {
  if (!isLoading) return null;

  return (
    <div
      className={clsx(
        'absolute inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-xl flex items-center justify-center z-20',
        className
      )}
    >
      <div className="text-center">
        <div className="relative inline-block">
          <Loader2 className="h-12 w-12 text-blue-600 dark:text-blue-400 animate-spin mx-auto mb-4" />
          <div className="absolute inset-0 bg-blue-600/20 dark:bg-blue-400/20 rounded-full blur-xl animate-pulse" />
        </div>
        <p className="text-lg font-semibold text-gray-700 dark:text-gray-300">{message}</p>
        {subMessage && (
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">{subMessage}</p>
        )}
        <div className="mt-4 flex items-center justify-center gap-1">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
          <div
            className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"
            style={{ animationDelay: '0.2s' }}
          />
          <div
            className="w-2 h-2 bg-pink-500 rounded-full animate-bounce"
            style={{ animationDelay: '0.4s' }}
          />
        </div>
      </div>
    </div>
  );
});

LoadingOverlay.displayName = 'LoadingOverlay';
