import { memo } from 'react';
import { useBattery } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Battery, BatteryCharging } from 'lucide-react';

interface BatteryIndicatorProps {
  className?: string;
  showPercentage?: boolean;
}

const BatteryIndicator = memo(({
  className = '',
  showPercentage = true,
}: BatteryIndicatorProps): JSX.Element => {
  const { level, charging, supported } = useBattery();

  if (!supported) {
    return null;
  }

  if (level === null) {
    return null;
  }

  const percentage = Math.round(level * 100);
  const getBatteryColor = (): string => {
    if (percentage > 50) return 'text-green-600';
    if (percentage > 20) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {charging ? (
        <BatteryCharging className={cn('w-5 h-5', getBatteryColor())} />
      ) : (
        <Battery className={cn('w-5 h-5', getBatteryColor())} />
      )}
      {showPercentage && (
        <span className={cn('text-sm font-medium', getBatteryColor())}>
          {percentage}%
        </span>
      )}
    </div>
  );
});

BatteryIndicator.displayName = 'BatteryIndicator';

export default BatteryIndicator;



