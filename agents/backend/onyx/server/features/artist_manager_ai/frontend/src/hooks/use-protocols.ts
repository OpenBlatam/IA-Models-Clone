import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { protocolsApi } from '@/lib/api-client';
import type { Protocol, ProtocolCompliance } from '@/types';

export const useProtocols = (
  artistId: string,
  params?: { category?: string; priority?: string; event_id?: string }
) => {
  return useQuery<Protocol[]>({
    queryKey: ['protocols', artistId, params],
    queryFn: () => protocolsApi.getProtocols(artistId, params),
    enabled: !!artistId,
  });
};

export const useProtocol = (artistId: string, protocolId: string) => {
  return useQuery<Protocol>({
    queryKey: ['protocol', artistId, protocolId],
    queryFn: () => protocolsApi.getProtocol(artistId, protocolId),
    enabled: !!artistId && !!protocolId,
  });
};

export const useCreateProtocol = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (protocol: Omit<Protocol, 'id'>) => protocolsApi.createProtocol(artistId, protocol),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['protocols', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useUpdateProtocol = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ protocolId, updates }: { protocolId: string; updates: Partial<Protocol> }) =>
      protocolsApi.updateProtocol(artistId, protocolId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['protocols', artistId] });
    },
  });
};

export const useDeleteProtocol = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (protocolId: string) => protocolsApi.deleteProtocol(artistId, protocolId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['protocols', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useCheckEventCompliance = (artistId: string) => {
  return useMutation({
    mutationFn: (eventId: string) => protocolsApi.checkEventCompliance(artistId, eventId),
  });
};

export const useRecordCompliance = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      protocolId,
      compliance,
    }: {
      protocolId: string;
      compliance: Omit<ProtocolCompliance, 'id' | 'checked_at'>;
    }) => protocolsApi.recordCompliance(artistId, protocolId, compliance),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['protocols', artistId] });
    },
  });
};

export const useComplianceRate = (artistId: string, protocolId: string) => {
  return useQuery<number>({
    queryKey: ['compliance-rate', artistId, protocolId],
    queryFn: () => protocolsApi.getComplianceRate(artistId, protocolId),
    enabled: !!artistId && !!protocolId,
  });
};

