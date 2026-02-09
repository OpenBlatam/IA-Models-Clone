import { memo } from 'react';
import Button from './Button';
import { cn } from '@/lib/utils';

interface RetryButtonProps {
  onRetry: () => void;
  isLoading?: boolean;
  className?: string;
  label?: string;
}

const RetryButton = memo(({
  onRetry,
  isLoading = false,
  className = '',
  label = 'Retry',
}: RetryButtonProps): JSX.Element => {
  return (
    <Button
      onClick={onRetry}
      isLoading={isLoading}
      variant="secondary"
      className={cn('w-full', className)}
      aria-label={label}
    >
      {label}
    </Button>
  );
});

RetryButton.displayName = 'RetryButton';

export default RetryButton;



