import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { wardrobeApi } from '@/lib/api-client';
import type { WardrobeItem, Outfit } from '@/types';

export const useWardrobeItems = (
  artistId: string,
  params?: { category?: string; dress_code?: string }
) => {
  return useQuery<WardrobeItem[]>({
    queryKey: ['wardrobe-items', artistId, params],
    queryFn: () => wardrobeApi.getItems(artistId, params),
    enabled: !!artistId,
  });
};

export const useWardrobeItem = (artistId: string, itemId: string) => {
  return useQuery<WardrobeItem>({
    queryKey: ['wardrobe-item', artistId, itemId],
    queryFn: () => wardrobeApi.getItem(artistId, itemId),
    enabled: !!artistId && !!itemId,
  });
};

export const useOutfits = (artistId: string, params?: { dress_code?: string }) => {
  return useQuery<Outfit[]>({
    queryKey: ['outfits', artistId, params],
    queryFn: () => wardrobeApi.getOutfits(artistId, params),
    enabled: !!artistId,
  });
};

export const useCreateWardrobeItem = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (item: Omit<WardrobeItem, 'id'>) => wardrobeApi.createItem(artistId, item),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wardrobe-items', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useUpdateWardrobeItem = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ itemId, updates }: { itemId: string; updates: Partial<WardrobeItem> }) =>
      wardrobeApi.updateItem(artistId, itemId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wardrobe-items', artistId] });
    },
  });
};

export const useDeleteWardrobeItem = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (itemId: string) => wardrobeApi.deleteItem(artistId, itemId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wardrobe-items', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useCreateOutfit = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (outfit: Omit<Outfit, 'id'>) => wardrobeApi.createOutfit(artistId, outfit),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['outfits', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useMarkItemWorn = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (itemId: string) => wardrobeApi.markItemWorn(artistId, itemId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wardrobe-items', artistId] });
    },
  });
};

export const useMarkOutfitWorn = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (outfitId: string) => wardrobeApi.markOutfitWorn(artistId, outfitId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['outfits', artistId] });
    },
  });
};

