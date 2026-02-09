'use client';

import React from 'react';
import { LazyLoad } from './LazyLoad';
import { ImageComponent } from './Image';
import { Skeleton } from './Skeleton';

interface LazyImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  fallback?: string;
}

export const LazyImage: React.FC<LazyImageProps> = ({
  src,
  alt,
  width,
  height,
  className,
  fallback,
}) => {
  return (
    <LazyLoad
      fallback={
        <Skeleton
          className={className}
          style={{ width, height }}
        />
      }
    >
      <ImageComponent
        src={src}
        alt={alt}
        width={width}
        height={height}
        className={className}
        fallback={fallback}
      />
    </LazyLoad>
  );
};
