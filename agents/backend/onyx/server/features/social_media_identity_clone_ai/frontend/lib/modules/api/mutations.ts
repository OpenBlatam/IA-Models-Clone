import { useMutation, useQueryClient } from 'react-query';
import { apiClient } from '@/lib/api/client';
import { queryKeys } from './queryKeys';
import { showToast } from '@/lib/integrations/react-hot-toast';
import type { ExtractProfileRequest, BuildIdentityRequest, GenerateContentRequest } from '@/types';

export const useExtractProfileMutation = () => {
  return useMutation(
    (request: ExtractProfileRequest) => apiClient.extractProfile(request),
    {
      onSuccess: () => {
        showToast.success('Profile extracted successfully!');
      },
      onError: () => {
        showToast.error('Failed to extract profile');
      },
    }
  );
};

export const useBuildIdentityMutation = () => {
  const queryClient = useQueryClient();

  return useMutation(
    (request: BuildIdentityRequest) => apiClient.buildIdentity(request),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries(queryKeys.identities.all);
        showToast.success('Identity built successfully!');
      },
      onError: () => {
        showToast.error('Failed to build identity');
      },
    }
  );
};

export const useGenerateContentMutation = () => {
  const queryClient = useQueryClient();

  return useMutation(
    (request: GenerateContentRequest) => apiClient.generateContent(request),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries(queryKeys.identities.generatedContent(data.content_id));
        showToast.success('Content generated successfully!');
      },
      onError: () => {
        showToast.error('Failed to generate content');
      },
    }
  );
};

export const useDeleteTemplateMutation = () => {
  const queryClient = useQueryClient();

  return useMutation(
    (templateId: string) => apiClient.deleteTemplate(templateId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(queryKeys.templates.all);
        showToast.success('Template deleted successfully');
      },
      onError: () => {
        showToast.error('Failed to delete template');
      },
    }
  );
};

export const useAcknowledgeAlertMutation = () => {
  const queryClient = useQueryClient();

  return useMutation(
    (alertId: string) => apiClient.acknowledgeAlert(alertId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(queryKeys.alerts.all);
        showToast.success('Alert acknowledged');
      },
      onError: () => {
        showToast.error('Failed to acknowledge alert');
      },
    }
  );
};

export const useResolveAlertMutation = () => {
  const queryClient = useQueryClient();

  return useMutation(
    (alertId: string) => apiClient.resolveAlert(alertId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(queryKeys.alerts.all);
        showToast.success('Alert resolved');
      },
      onError: () => {
        showToast.error('Failed to resolve alert');
      },
    }
  );
};



