import PageLayout from './PageLayout';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import Card from '@/components/UI/Card';
import EmptyState from '@/components/UI/EmptyState';

interface PageWrapperProps {
  isLoading: boolean;
  error?: Error | null;
  isEmpty?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
  emptyActionLabel?: string;
  onEmptyAction?: () => void;
  children: React.ReactNode;
}

const PageWrapper = ({
  isLoading,
  error,
  isEmpty = false,
  emptyTitle,
  emptyDescription,
  emptyActionLabel,
  onEmptyAction,
  children,
}: PageWrapperProps): JSX.Element => {
  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  if (error) {
    return (
      <PageLayout>
        <Card role="alert" aria-live="polite">
          <div className="text-red-600">
            <p className="font-semibold mb-2">Error:</p>
            <p>{error.message || 'An error occurred'}</p>
          </div>
        </Card>
      </PageLayout>
    );
  }

  if (isEmpty) {
    return (
      <PageLayout>
        <Card>
          <EmptyState
            title={emptyTitle || 'No data available'}
            description={emptyDescription}
            actionLabel={emptyActionLabel}
            onAction={onEmptyAction}
          />
        </Card>
      </PageLayout>
    );
  }

  return <PageLayout>{children}</PageLayout>;
};

export default PageWrapper;



