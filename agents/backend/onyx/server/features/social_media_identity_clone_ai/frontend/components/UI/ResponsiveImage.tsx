import { memo } from 'react';
import { cn } from '@/lib/utils';

interface ResponsiveImageProps {
  src: string;
  alt: string;
  fallback?: string;
  className?: string;
  sizes?: string;
  srcSet?: string;
}

const ResponsiveImage = memo(({
  src,
  alt,
  fallback,
  className = '',
  sizes,
  srcSet,
}: ResponsiveImageProps): JSX.Element => {
  return (
    <img
      src={src}
      alt={alt}
      className={cn('w-full h-auto', className)}
      sizes={sizes}
      srcSet={srcSet}
      loading="lazy"
      decoding="async"
      onError={(e) => {
        if (fallback && e.currentTarget.src !== fallback) {
          e.currentTarget.src = fallback;
        }
      }}
    />
  );
});

ResponsiveImage.displayName = 'ResponsiveImage';

export default ResponsiveImage;

