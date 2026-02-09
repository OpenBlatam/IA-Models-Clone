import { memo } from 'react';
import Card from '@/components/UI/Card';
import { getErrorMessage } from '@/lib/utils';

interface MutationErrorProps {
  error: unknown;
  className?: string;
}

const MutationError = memo(({ error, className = '' }: MutationErrorProps): JSX.Element => {
  if (!error) {
    return null;
  }

  return (
    <Card className={`mt-6 border-red-500 ${className}`} role="alert" aria-live="polite">
      <div className="text-red-600">
        <p className="font-semibold">Error:</p>
        <p>{getErrorMessage(error)}</p>
      </div>
    </Card>
  );
});

MutationError.displayName = 'MutationError';

export default MutationError;



