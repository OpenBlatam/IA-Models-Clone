'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { usePagination } from '@/lib/hooks/use-pagination';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { ManualList } from './manual/manual-list';
import { Pagination } from './ui/pagination';
import { pluralize } from '@/lib/utils/pluralize';
import { DEFAULT_USER_ID, MESSAGES } from '@/lib/constants';

export const FavoritesList = (): JSX.Element => {
  const { limit, offset, currentPage, nextPage, previousPage } = usePagination();

  const { data: favorites, isLoading, error } = useQuery({
    queryKey: ['favorites', DEFAULT_USER_ID, limit, offset],
    queryFn: () => apiClient.getUserFavorites(DEFAULT_USER_ID, limit, offset),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Favoritos</CardTitle>
        <CardDescription>
          {favorites
            ? `${favorites.length} ${pluralize(favorites.length, 'manual', 'manuales')} en favoritos`
            : MESSAGES.EMPTY.NO_FAVORITES}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ManualList
          manuals={favorites}
          isLoading={isLoading}
          error={error}
          emptyMessage={MESSAGES.EMPTY.NO_FAVORITES}
        />

        {favorites && favorites.length > 0 && (
          <Pagination
            currentPage={currentPage}
            totalItems={favorites.length}
            itemsPerPage={limit}
            onPrevious={previousPage}
            onNext={nextPage}
            hasMore={favorites.length >= limit}
          />
        )}
      </CardContent>
    </Card>
  );
};

