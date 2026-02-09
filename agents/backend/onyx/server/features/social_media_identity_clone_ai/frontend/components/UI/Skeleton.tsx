import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
}

const Skeleton = ({ className = '', variant = 'rectangular' }: SkeletonProps): JSX.Element => {
  const baseClasses = 'animate-pulse bg-gray-200';
  
  const variantClasses = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded',
  };

  return (
    <div
      className={cn(baseClasses, variantClasses[variant], className)}
      aria-label="Loading..."
      role="status"
    />
  );
};

export default Skeleton;



