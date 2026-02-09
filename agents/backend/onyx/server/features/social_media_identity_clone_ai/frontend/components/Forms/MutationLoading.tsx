import { memo } from 'react';
import Card from '@/components/UI/Card';
import LoadingSpinner from '@/components/UI/LoadingSpinner';

interface MutationLoadingProps {
  isLoading: boolean;
  message?: string;
  className?: string;
}

const MutationLoading = memo(({ isLoading, message, className = '' }: MutationLoadingProps): JSX.Element => {
  if (!isLoading) {
    return null;
  }

  return (
    <Card className={`mt-6 ${className}`}>
      <LoadingSpinner />
      {message && <p className="text-center text-gray-600 mt-4">{message}</p>}
    </Card>
  );
});

MutationLoading.displayName = 'MutationLoading';

export default MutationLoading;



