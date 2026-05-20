import { useQueryClient } from '@tanstack/react-query';

export const useQueryInvalidation = () => {
  const queryClient = useQueryClient();

  const invalidateManuals = (): void => {
    queryClient.invalidateQueries({ queryKey: ['manuals'] });
    queryClient.invalidateQueries({ queryKey: ['recent-manuals'] });
  };

  const invalidateManual = (manualId: number): void => {
    queryClient.invalidateQueries({ queryKey: ['manual', manualId] });
    queryClient.invalidateQueries({ queryKey: ['ratings', manualId] });
  };

  const invalidateFavorites = (userId: string): void => {
    queryClient.invalidateQueries({ queryKey: ['favorites', userId] });
  };

  const invalidateAll = (): void => {
    invalidateManuals();
    queryClient.invalidateQueries({ queryKey: ['statistics'] });
  };

  return {
    invalidateManuals,
    invalidateManual,
    invalidateFavorites,
    invalidateAll,
  };
};

