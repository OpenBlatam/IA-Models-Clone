import { memo } from 'react';
import { cn } from '@/lib/utils';

interface MetricItemProps {
  label: string;
  value: string | number;
  className?: string;
}

const MetricItem = memo(({ label, value, className = '' }: MetricItemProps): JSX.Element => {
  return (
    <div className={cn('flex items-center justify-between py-2 border-b border-gray-100', className)}>
      <span className="text-sm text-gray-600">{label}</span>
      <span className="text-sm font-semibold text-gray-900">{value}</span>
    </div>
  );
});

MetricItem.displayName = 'MetricItem';

export default MetricItem;
