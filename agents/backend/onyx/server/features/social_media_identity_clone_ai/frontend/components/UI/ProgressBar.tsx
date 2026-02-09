import { cn } from '@/lib/utils';

interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  className?: string;
  variant?: 'primary' | 'success' | 'warning' | 'error';
}

const VARIANT_CLASSES = {
  primary: 'bg-primary-600',
  success: 'bg-green-600',
  warning: 'bg-yellow-600',
  error: 'bg-red-600',
};

const ProgressBar = ({
  value,
  max = 100,
  label,
  showPercentage = false,
  className = '',
  variant = 'primary',
}: ProgressBarProps): JSX.Element => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const variantClass = VARIANT_CLASSES[variant];

  return (
    <div className={cn('w-full', className)}>
      {(label || showPercentage) && (
        <div className="flex items-center justify-between mb-2">
          {label && <span className="text-sm text-gray-600">{label}</span>}
          {showPercentage && (
            <span className="text-sm font-medium text-gray-900">{Math.round(percentage)}%</span>
          )}
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={cn('h-full transition-all duration-300', variantClass)}
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={value}
          aria-valuemin={0}
          aria-valuemax={max}
          aria-label={label || `Progress: ${percentage}%`}
        />
      </div>
    </div>
  );
};

export default ProgressBar;



