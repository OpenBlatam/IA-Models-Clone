import { memo } from 'react';
import { useFullscreen } from '@/lib/hooks';
import Button from './Button';
import { Maximize, Minimize } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FullscreenButtonProps {
  className?: string;
  variant?: 'primary' | 'secondary' | 'danger';
}

const FullscreenButton = memo(({
  className = '',
  variant = 'secondary',
}: FullscreenButtonProps): JSX.Element => {
  const { isFullscreen, toggleFullscreen } = useFullscreen();

  return (
    <Button
      onClick={toggleFullscreen}
      variant={variant}
      className={cn('flex items-center gap-2', className)}
      aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
    >
      {isFullscreen ? (
        <>
          <Minimize className="w-4 h-4" />
          Exit Fullscreen
        </>
      ) : (
        <>
          <Maximize className="w-4 h-4" />
          Fullscreen
        </>
      )}
    </Button>
  );
});

FullscreenButton.displayName = 'FullscreenButton';

export default FullscreenButton;



