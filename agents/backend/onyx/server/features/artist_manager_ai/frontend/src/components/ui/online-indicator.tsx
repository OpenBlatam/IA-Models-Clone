'use client';

import { useOnlineStatus } from '@/hooks/use-online-status';
import { Alert } from '@/components/ui/alert';
import { Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';

interface OnlineIndicatorProps {
  showWhenOnline?: boolean;
  className?: string;
  position?: 'top' | 'bottom';
}

const OnlineIndicator = ({
  showWhenOnline = false,
  className,
  position = 'top',
}: OnlineIndicatorProps) => {
  const isOnline = useOnlineStatus();

  if (isOnline && !showWhenOnline) {
    return null;
  }

  return (
    <div
      className={cn(
        'fixed left-0 right-0 z-50 px-4',
        position === 'top' ? 'top-0' : 'bottom-0',
        className
      )}
    >
      <Alert
        variant={isOnline ? 'success' : 'error'}
        message={
          <div className="flex items-center gap-2">
            {isOnline ? (
              <>
                <Wifi className="w-4 h-4" />
                <span>Conexión restaurada</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4" />
                <span>Sin conexión a internet</span>
              </>
            )}
          </div>
        }
        className="mb-0"
      />
    </div>
  );
};

export { OnlineIndicator };

