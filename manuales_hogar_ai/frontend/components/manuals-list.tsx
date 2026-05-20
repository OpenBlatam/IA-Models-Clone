'use client';

import { useManuals } from '@/lib/hooks/use-manuals';
import { usePagination } from '@/lib/hooks/use-pagination';
import { useSearchState } from '@/lib/hooks/use-search-state';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { CategorySelect } from './manual/category-select';
import { ManualList } from './manual/manual-list';
import { Pagination } from './ui/pagination';
import { SearchInput } from './search/search-input';
import { MESSAGES } from '@/lib/constants';

export const ManualsList = (): JSX.Element => {
  const { query, category, setQuery, setCategory } = useSearchState();
  const { limit, offset, currentPage, nextPage, previousPage, reset: resetPagination } = usePagination();

  const { data: manuals, isLoading, error } = useManuals(category, query, limit, offset);

  const handleSearch = (): void => {
    resetPagination();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Manuales</CardTitle>
        <CardDescription>
          Busca y filtra manuales generados
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <SearchInput
                value={query}
                onChange={setQuery}
                onSearch={handleSearch}
                placeholder="Buscar manuales..."
                ariaLabel="Buscar manuales"
              />
            </div>
            <div className="w-48">
              <CategorySelect
                value={category || 'all'}
                onValueChange={setCategory}
                includeAll
              />
            </div>
          </div>

          <ManualList
            manuals={manuals}
            isLoading={isLoading && offset === 0}
            error={error}
            emptyMessage={MESSAGES.EMPTY.NO_MANUALS}
          />

          {manuals && manuals.length > 0 && (
            <Pagination
              currentPage={currentPage}
              totalItems={manuals.length}
              itemsPerPage={limit}
              onPrevious={previousPage}
              onNext={nextPage}
              hasMore={manuals.length >= limit}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

