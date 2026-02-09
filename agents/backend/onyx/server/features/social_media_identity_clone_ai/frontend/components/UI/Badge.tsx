import { cn } from '@/lib/utils';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error';
  className?: string;
}

const Badge = ({ children, variant = 'default', className }: BadgeProps): JSX.Element => {
  const variantClasses = {
    default: 'bg-gray-100 text-gray-700',
    primary: 'bg-primary-100 text-primary-700',
    success: 'bg-green-100 text-green-700',
    warning: 'bg-yellow-100 text-yellow-700',
    error: 'bg-red-100 text-red-700',
  };

  return (
    <span className={cn('px-2 py-1 rounded-full text-xs font-medium', variantClasses[variant], className)}>
      {children}
    </span>
  );
};

export default Badge;



