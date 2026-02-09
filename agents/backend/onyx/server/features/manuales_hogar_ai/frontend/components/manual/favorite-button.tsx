'use client';

import { useState } from 'react';
import { Button } from '../ui/button';
import { Heart, Loader2 } from 'lucide-react';
import { useAddFavorite, useRemoveFavorite, useCheckFavorite } from '@/lib/hooks/use-manuals';
import { showErrorToast, showSuccessToast } from '@/lib/utils/error-handler';
import { DEFAULT_USER_ID, MESSAGES } from '@/lib/constants';
import type { FavoriteButtonProps } from '@/lib/types/components';

export const FavoriteButton = ({
  manualId,
  userId = DEFAULT_USER_ID,
  variant = 'outline',
  size = 'default',
}: FavoriteButtonProps): JSX.Element => {
  const { data: favoriteData } = useCheckFavorite(manualId, userId);
  const addFavorite = useAddFavorite();
  const removeFavorite = useRemoveFavorite();

  const isFavorite = favoriteData?.is_favorite || false;
  const isLoading = addFavorite.isPending || removeFavorite.isPending;

  const handleToggleFavorite = async (): Promise<void> => {
    try {
      if (isFavorite) {
        await removeFavorite.mutateAsync({ manualId, userId });
        showSuccessToast(MESSAGES.FAVORITE.REMOVE_SUCCESS);
      } else {
        await addFavorite.mutateAsync({ manualId, userId });
        showSuccessToast(MESSAGES.FAVORITE.ADD_SUCCESS);
      }
    } catch (error) {
      showErrorToast(error);
    }
  };

  return (
    <Button
      variant={isFavorite ? 'default' : variant}
      onClick={handleToggleFavorite}
      disabled={isLoading}
      size={size}
      aria-label={isFavorite ? 'Remover de favoritos' : 'Agregar a favoritos'}
    >
      {isLoading ? (
        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
      ) : (
        <Heart className={`h-4 w-4 mr-2 ${isFavorite ? 'fill-current' : ''}`} />
      )}
      {size !== 'icon' && (isFavorite ? 'En Favoritos' : 'Agregar a Favoritos')}
    </Button>
  );
};

