/**
 * Custom hook for social media connections
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { connectionsApi } from '@/lib/api/connections';
import type { SocialMediaConnectRequest, SocialMediaPlatform } from '@/lib/types';
import toast from 'react-hot-toast';

export const useConnections = () => {
  return useQuery({
    queryKey: ['connections'],
    queryFn: () => connectionsApi.getAll(),
  });
};

export const useConnectSocialMedia = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: SocialMediaConnectRequest) => connectionsApi.connect(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['connections'] });
      toast.success('Red social conectada exitosamente');
    },
    onError: (error: Error) => {
      toast.error(`Error al conectar: ${error.message}`);
    },
  });
};

export const useDisconnectSocialMedia = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (platform: SocialMediaPlatform) => connectionsApi.disconnect(platform),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['connections'] });
      toast.success('Red social desconectada');
    },
    onError: (error: Error) => {
      toast.error(`Error al desconectar: ${error.message}`);
    },
  });
};




