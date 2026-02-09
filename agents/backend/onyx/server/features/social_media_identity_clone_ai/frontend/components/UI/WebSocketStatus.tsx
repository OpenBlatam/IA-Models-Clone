import { memo } from 'react';
import { useWebSocket } from '@/lib/hooks';
import Badge from './Badge';
import { cn } from '@/lib/utils';
import { Wifi, WifiOff } from 'lucide-react';

interface WebSocketStatusProps {
  url: string | null;
  className?: string;
  showIcon?: boolean;
}

const WebSocketStatus = memo(({
  url,
  className = '',
  showIcon = true,
}: WebSocketStatusProps): JSX.Element => {
  const { isConnected } = useWebSocket(url);

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showIcon && (
        <>
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-600" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-600" />
          )}
        </>
      )}
      <Badge variant={isConnected ? 'success' : 'danger'}>
        {isConnected ? 'Connected' : 'Disconnected'}
      </Badge>
    </div>
  );
});

WebSocketStatus.displayName = 'WebSocketStatus';

export default WebSocketStatus;



