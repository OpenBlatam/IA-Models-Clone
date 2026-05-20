/**
 * Optimized track image component.
 * Uses Next.js Image component for automatic optimization, lazy loading, and responsive images.
 */

import Image from 'next/image';
import { Music } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TrackImageProps {
  src?: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  priority?: boolean;
  sizes?: string;
}

/**
 * Optimized track image component.
 * Automatically handles image optimization, lazy loading, and responsive sizing.
 *
 * @param props - Component props
 * @returns Optimized image component
 */
export function TrackImage({
  src,
  alt,
  width = 48,
  height = 48,
  className,
  priority = false,
  sizes = '(max-width: 768px) 48px, 64px',
}: TrackImageProps) {
  // Fallback to placeholder if no image
  if (!src) {
    return (
      <div
        className={cn(
          'bg-purple-500 flex items-center justify-center rounded',
          className
        )}
        style={{ width, height }}
        aria-label={alt}
      >
        <Music className="w-6 h-6 text-white" aria-hidden="true" />
      </div>
    );
  }

  return (
    <div className={cn('relative overflow-hidden rounded', className)}>
      <Image
        src={src}
        alt={alt}
        width={width}
        height={height}
        className="object-cover"
        loading={priority ? 'eager' : 'lazy'}
        sizes={sizes}
        placeholder="blur"
        blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
      />
    </div>
  );
}

