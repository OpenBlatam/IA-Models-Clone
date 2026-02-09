'use client';

import { memo } from 'react';
import Image from 'next/image';
import { useImageLoader } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Skeleton } from './Skeleton';
import { AlertCircle } from 'lucide-react';

interface ImageProps {
  src: string | null | undefined;
  alt: string;
  width?: number;
  height?: number;
  fill?: boolean;
  className?: string;
  objectFit?: 'contain' | 'cover' | 'fill' | 'none' | 'scale-down';
  onLoad?: () => void;
  onError?: (error: Error) => void;
  fallback?: React.ReactNode;
}

const CustomImage = memo(
  ({
    src,
    alt,
    width,
    height,
    fill = false,
    className,
    objectFit = 'cover',
    onLoad,
    onError,
    fallback,
  }: ImageProps): JSX.Element => {
    const { isLoading, hasError, imageSrc } = useImageLoader({
      src,
      onLoad,
      onError,
    });

    if (hasError) {
      return (
        <div
          className={cn(
            'flex items-center justify-center bg-gray-100 text-gray-400',
            className
          )}
          style={fill ? undefined : { width, height }}
        >
          {fallback || (
            <div className="text-center">
              <AlertCircle className="w-8 h-8 mx-auto mb-2" aria-hidden="true" />
              <p className="text-sm">Failed to load image</p>
            </div>
          )}
        </div>
      );
    }

    if (isLoading || !imageSrc) {
      return (
        <Skeleton
          className={cn(className)}
          style={fill ? undefined : { width, height }}
        />
      );
    }

    return (
      <Image
        src={imageSrc}
        alt={alt}
        width={width}
        height={height}
        fill={fill}
        className={cn(className)}
        style={{ objectFit }}
        unoptimized
      />
    );
  }
);

CustomImage.displayName = 'CustomImage';

export default CustomImage;

