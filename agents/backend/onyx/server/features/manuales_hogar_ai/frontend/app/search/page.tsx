import { PageContainer } from '@/components/layout/page-container';
import { PageHeader } from '@/components/layout/page-header';
import { SearchPanel } from '@/components/search-panel';

const SearchPage = (): JSX.Element => {
  return (
    <PageContainer>
      <div className="max-w-6xl mx-auto">
        <PageHeader
          title="Búsqueda de Manuales"
          description="Busca manuales usando diferentes métodos de búsqueda"
        />
        <SearchPanel />
      </div>
    </PageContainer>
  );
};

export default SearchPage;

