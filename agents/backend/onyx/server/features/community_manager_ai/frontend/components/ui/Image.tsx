'use client';

import { useState, ImgHTMLAttributes } from 'react';
import { ImageIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface ImageProps extends Omit<ImgHTMLAttributes<HTMLImageElement>, 'onError' | 'onLoad'> {
  fallback?: string;
  showFallback?: boolean;
  className?: string;
}

export const Image = ({
  src,
  alt,
  fallback,
  showFallback = true,
  className,
  ...props
}: ImageProps) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  const handleLoad = () => {
    setIsLoaded(true);
  };

  const handleError = () => {
    setHasError(true);
  };

  if (hasError && showFallback) {
    return (
      <div
        className={cn(
          'flex items-center justify-center bg-gray-100 dark:bg-gray-800',
          className
        )}
        role="img"
        aria-label={alt || 'Imagen no disponible'}
      >
        {fallback ? (
          <img src={fallback} alt={alt} className="h-full w-full object-cover" />
        ) : (
          <ImageIcon className="h-8 w-8 text-gray-400 dark:text-gray-500" />
        )}
      </div>
    );
  }

  return (
    <div className={cn('relative overflow-hidden', className)}>
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
          <ImageIcon className="h-8 w-8 text-gray-400 dark:text-gray-500 animate-pulse" />
        </div>
      )}
      {/* @ts-ignore - framer-motion types conflict */}
      <motion.img
        src={src}
        alt={alt}
        onLoad={handleLoad}
        onError={handleError}
        initial={{ opacity: 0 }}
        animate={{ opacity: isLoaded ? 1 : 0 }}
        transition={{ duration: 0.3 }}
        className={cn('h-full w-full object-cover', !isLoaded && 'invisible')}
        {...props}
      />
    </div>
  );
};

