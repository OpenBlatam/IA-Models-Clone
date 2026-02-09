import { memo, useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import Badge from './Badge';

const ConnectionStatus = memo((): JSX.Element => {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    setIsOnline(navigator.onLine);

    const handleOnline = (): void => setIsOnline(true);
    const handleOffline = (): void => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (isOnline) {
    return null;
  }

  return (
    <div
      className={cn(
        'fixed bottom-4 right-4 z-50',
        'bg-yellow-100 border border-yellow-400 text-yellow-700',
        'px-4 py-2 rounded-lg shadow-lg',
        'flex items-center gap-2'
      )}
      role="alert"
      aria-live="assertive"
    >
      <Badge variant="warning" className="text-xs">
        Offline
      </Badge>
      <span className="text-sm">No internet connection</span>
    </div>
  );
});

ConnectionStatus.displayName = 'ConnectionStatus';

export default ConnectionStatus;



