import { PageContainer } from '@/components/layout/page-container';
import { PageHeader } from '@/components/layout/page-header';
import { ManualsList } from '@/components/manuals-list';

const HistoryPage = (): JSX.Element => {
  return (
    <PageContainer>
      <div className="max-w-6xl mx-auto">
        <PageHeader
          title="Historial de Manuales"
          description="Busca y filtra todos tus manuales generados"
        />
        <ManualsList />
      </div>
    </PageContainer>
  );
};

export default HistoryPage;

