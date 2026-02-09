import { memo } from 'react';
import { cn } from '@/lib/utils';

interface InfoRowProps {
  label: string;
  value: React.ReactNode;
  className?: string;
  labelClassName?: string;
  valueClassName?: string;
}

const InfoRow = memo(({
  label,
  value,
  className = '',
  labelClassName = '',
  valueClassName = '',
}: InfoRowProps): JSX.Element => {
  return (
    <div className={cn('space-y-1', className)}>
      <p className={cn('text-sm text-gray-600', labelClassName)}>{label}</p>
      <div className={cn('text-gray-900', valueClassName)}>{value}</div>
    </div>
  );
});

InfoRow.displayName = 'InfoRow';

export default InfoRow;



