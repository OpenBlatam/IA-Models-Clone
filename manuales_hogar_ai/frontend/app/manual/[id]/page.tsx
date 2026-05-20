import { PageContainer } from '@/components/layout/page-container';
import { ErrorState } from '@/components/ui/error-state';
import { ManualDetail } from '@/components/manual-detail';
import { MESSAGES } from '@/lib/constants';

interface ManualPageProps {
  params: {
    id: string;
  };
}

const ManualPage = ({ params }: ManualPageProps): JSX.Element => {
  const manualId = parseInt(params.id, 10);

  if (isNaN(manualId)) {
    return (
      <PageContainer>
        <div className="max-w-4xl mx-auto">
          <ErrorState title="Error" message={MESSAGES.MANUAL.INVALID_ID} />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <div className="max-w-4xl mx-auto">
        <ManualDetail manualId={manualId} />
      </div>
    </PageContainer>
  );
};

export default ManualPage;

