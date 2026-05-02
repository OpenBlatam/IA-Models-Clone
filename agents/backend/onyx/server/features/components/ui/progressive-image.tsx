"use client";

import React, { useState } from "react";
import Image from "next/image";
import { usePerformance } from "@/components/providers/performance-provider";
import { cn } from "@/lib/utils";

interface ProgressiveImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  priority?: boolean;
}

export function ProgressiveImage({
  src,
  alt,
  width,
  height,
  className,
  priority = false,
  ...props
}: ProgressiveImageProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const { imageQuality, isSlowConnection } = usePerformance();
  
  const getOptimizedSrc = (originalSrc: string) => {
    if (isSlowConnection && imageQuality === 'low') {
      return originalSrc.replace(/\.(jpg|jpeg|png)$/i, '_low.$1');
    }
    return originalSrc;
  };

  return (
    <div className={cn("relative overflow-hidden", className)}>
      <Image
        src={getOptimizedSrc(src)}
        alt={alt}
        width={width}
        height={height}
        priority={priority}
        quality={imageQuality === 'low' ? 50 : imageQuality === 'medium' ? 75 : 90}
        onLoad={() => setIsLoaded(true)}
        className={cn(
          "transition-opacity duration-300",
          isLoaded ? "opacity-100" : "opacity-0"
        )}
        {...props}
      />
      {!isLoaded && (
        <div className="absolute inset-0 bg-muted animate-pulse" />
      )}
    </div>
  );
}
