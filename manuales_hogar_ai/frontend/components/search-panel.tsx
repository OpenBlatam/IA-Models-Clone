'use client';

import { useSearch } from '@/lib/hooks/use-manuals';
import { useSemanticSearch, useAdvancedSearch } from '@/lib/hooks/use-search';
import { useSearchState } from '@/lib/hooks/use-search-state';
import { useTabs } from '@/lib/hooks/use-tabs';
import { useDebounce } from '@/lib/hooks/use-debounce';
import { useSearchResults } from '@/lib/hooks/use-search-results';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { SearchInput } from './search/search-input';
import { ManualList } from './manual/manual-list';
import { SEARCH, SEARCH_TABS, MESSAGES } from '@/lib/constants';

export const SearchPanel = (): JSX.Element => {
  const { query, category, setQuery, setCategory } = useSearchState();
  const { activeTab, setActiveTab } = useTabs({ defaultValue: SEARCH_TABS.SIMPLE });
  const debouncedQuery = useDebounce(query, SEARCH.DEBOUNCE_MS);
  const limit = SEARCH.DEFAULT_LIMIT;

  const simpleSearch = useSearch(debouncedQuery, category, limit, 0);
  const semanticSearch = useSemanticSearch(debouncedQuery, category, limit, 0.5);
  const advancedSearch = useAdvancedSearch({
    query: debouncedQuery,
    category,
    limit,
    offset: 0,
  });

  const handleSearch = (): void => {
    if (query.trim().length === 0) return;
  };

  const { results: normalizedResults, isLoading, error } = useSearchResults({
    simpleSearch,
    semanticSearch,
    advancedSearch,
    activeTab,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Búsqueda de Manuales</CardTitle>
        <CardDescription>
          Busca manuales usando diferentes métodos de búsqueda
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value={SEARCH_TABS.SIMPLE}>Búsqueda Simple</TabsTrigger>
            <TabsTrigger value={SEARCH_TABS.SEMANTIC}>Búsqueda Semántica</TabsTrigger>
            <TabsTrigger value={SEARCH_TABS.ADVANCED}>Búsqueda Avanzada</TabsTrigger>
          </TabsList>

          <TabsContent value={SEARCH_TABS.SIMPLE} className="space-y-4">
            <SearchInput
              value={query}
              onChange={setQuery}
              onSearch={handleSearch}
              placeholder="Buscar manuales..."
              ariaLabel="Buscar manuales"
            />
            <ManualList
              manuals={normalizedResults}
              isLoading={isLoading}
              error={error}
              emptyMessage={query ? MESSAGES.SEARCH.NO_RESULTS : undefined}
            />
          </TabsContent>

          <TabsContent value={SEARCH_TABS.SEMANTIC} className="space-y-4">
            <SearchInput
              value={query}
              onChange={setQuery}
              onSearch={handleSearch}
              placeholder="Buscar por significado..."
              ariaLabel="Buscar por significado"
            />
            <ManualList
              manuals={normalizedResults}
              isLoading={isLoading}
              error={error}
              emptyMessage={query ? MESSAGES.SEARCH.NO_RESULTS : undefined}
            />
          </TabsContent>

          <TabsContent value={SEARCH_TABS.ADVANCED} className="space-y-4">
            <SearchInput
              value={query}
              onChange={setQuery}
              onSearch={handleSearch}
              placeholder="Búsqueda avanzada..."
              ariaLabel="Búsqueda avanzada"
            />
            <ManualList
              manuals={normalizedResults}
              isLoading={isLoading}
              error={error}
              emptyMessage={query ? MESSAGES.SEARCH.NO_RESULTS : undefined}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

