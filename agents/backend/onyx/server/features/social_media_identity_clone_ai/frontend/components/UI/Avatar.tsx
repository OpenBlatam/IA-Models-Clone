import { cn } from '@/lib/utils';
import Image from './Image';

interface AvatarProps {
  src?: string;
  alt: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  fallback?: string;
}

const SIZE_CLASSES = {
  sm: 'w-8 h-8 text-sm',
  md: 'w-10 h-10 text-base',
  lg: 'w-12 h-12 text-lg',
  xl: 'w-16 h-16 text-xl',
};

const Avatar = ({ src, alt, size = 'md', className = '', fallback }: AvatarProps): JSX.Element => {
  const sizeClass = SIZE_CLASSES[size];
  const initials = alt
    .split(' ')
    .map((word) => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  if (!src && !fallback) {
    return (
      <div
        className={cn(
          'rounded-full bg-primary-600 text-white flex items-center justify-center font-semibold',
          sizeClass,
          className
        )}
        role="img"
        aria-label={alt}
      >
        {initials}
      </div>
    );
  }

  return (
    <div className={cn('rounded-full overflow-hidden', sizeClass, className)}>
      <Image
        src={src || fallback || ''}
        alt={alt}
        fallback={fallback}
        className="w-full h-full"
      />
    </div>
  );
};

export default Avatar;



