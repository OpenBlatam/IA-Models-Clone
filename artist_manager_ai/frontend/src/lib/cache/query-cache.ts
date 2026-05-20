import { QueryClient } from '@tanstack/react-query';
import { queryKeys } from '../query-keys';

export const invalidateArtistQueries = (queryClient: QueryClient, artistId: string) => {
  // Invalidate all queries for this artist
  queryClient.invalidateQueries({
    predicate: (query) => {
      const key = query.queryKey;
      return Array.isArray(key) && key.includes(artistId);
    },
  });
};

export const prefetchDashboard = async (queryClient: QueryClient, artistId: string) => {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.dashboard(artistId),
  });
};

export const prefetchEvents = async (queryClient: QueryClient, artistId: string, params?: Record<string, any>) => {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.events(artistId, params),
  });
};

export const clearArtistCache = (queryClient: QueryClient, artistId: string) => {
  queryClient.removeQueries({
    predicate: (query) => {
      const key = query.queryKey;
      return Array.isArray(key) && key.includes(artistId);
    },
  });
};

