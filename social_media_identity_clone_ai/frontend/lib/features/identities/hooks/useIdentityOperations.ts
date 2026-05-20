import { useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useBuildIdentityMutation, useIdentities } from '@/lib/modules/api';
import { showToast } from '@/lib/integrations/react-hot-toast';
import type { BuildIdentityRequest } from '@/types';

export const useIdentityOperations = () => {
  const router = useRouter();
  const buildMutation = useBuildIdentityMutation();
  const { refetch: refetchIdentities } = useIdentities();

  const handleBuildIdentity = useCallback(
    async (request: BuildIdentityRequest) => {
      return buildMutation.mutateAsync(request, {
        onSuccess: async (data) => {
          await refetchIdentities();
          router.push(`/identities/${data.identity_id}`);
        },
      });
    },
    [buildMutation, refetchIdentities, router]
  );

  const handleNavigateToIdentity = useCallback(
    (identityId: string) => {
      router.push(`/identities/${identityId}`);
    },
    [router]
  );

  const handleNavigateToBuild = useCallback(() => {
    router.push('/build-identity');
  }, [router]);

  return {
    buildIdentity: handleBuildIdentity,
    navigateToIdentity: handleNavigateToIdentity,
    navigateToBuild: handleNavigateToBuild,
    isBuilding: buildMutation.isLoading,
    buildError: buildMutation.error,
  };
};



