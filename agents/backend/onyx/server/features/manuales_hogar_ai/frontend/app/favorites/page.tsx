import { PageContainer } from '@/components/layout/page-container';
import { PageHeader } from '@/components/layout/page-header';
import { FavoritesList } from '@/components/favorites-list';

const FavoritesPage = (): JSX.Element => {
  return (
    <PageContainer>
      <div className="max-w-6xl mx-auto">
        <PageHeader
          title="Mis Favoritos"
          description="Manuales que has marcado como favoritos"
        />
        <FavoritesList />
      </div>
    </PageContainer>
  );
};

export default FavoritesPage;

