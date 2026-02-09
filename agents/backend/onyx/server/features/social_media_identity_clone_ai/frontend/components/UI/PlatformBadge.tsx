import { memo } from 'react';
import { Platform } from '@/types';
import { getPlatformColor, getPlatformIcon } from '@/lib/utils';
import { cn } from '@/lib/utils';

interface PlatformBadgeProps {
  platform: Platform;
  showIcon?: boolean;
  className?: string;
}

const PlatformBadge = memo(({ platform, showIcon = true, className = '' }: PlatformBadgeProps): JSX.Element => {
  const color = getPlatformColor(platform);
  const Icon = getPlatformIcon(platform);

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
        color,
        className
      )}
      aria-label={`Platform: ${platform}`}
    >
      {showIcon && Icon && <Icon className="w-3 h-3" />}
      <span className="capitalize">{platform}</span>
    </span>
  );
});

PlatformBadge.displayName = 'PlatformBadge';

export default PlatformBadge;
