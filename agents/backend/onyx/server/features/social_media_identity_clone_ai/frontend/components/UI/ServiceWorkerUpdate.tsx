import { memo } from 'react';
import { useServiceWorker } from '@/lib/hooks';
import Button from './Button';
import Card from './Card';
import { RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ServiceWorkerUpdateProps {
  className?: string;
}

const ServiceWorkerUpdate = memo(({ className = '' }: ServiceWorkerUpdateProps): JSX.Element => {
  const { updateAvailable, update, supported } = useServiceWorker();

  if (!supported || !updateAvailable) {
    return null;
  }

  return (
    <Card className={cn('fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 z-50', className)}>
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="font-semibold mb-1">Update Available</h3>
          <p className="text-sm text-gray-600">A new version is available. Click to update.</p>
        </div>
        <Button onClick={update} variant="primary" className="flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Update
        </Button>
      </div>
    </Card>
  );
});

ServiceWorkerUpdate.displayName = 'ServiceWorkerUpdate';

export default ServiceWorkerUpdate;



