'use client';

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useParams, useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import ConfirmDialog from '@/components/UI/ConfirmDialog';
import { formatDate } from '@/lib/utils';
import { useState } from 'react';

interface Version {
  version_id: string;
  created_at: string;
  description?: string;
}

const VersionsPage = (): JSX.Element => {
  const params = useParams();
  const router = useRouter();
  const identityId = params.id as string;
  const [restoreConfirm, setRestoreConfirm] = useState<{ isOpen: boolean; version: string | null }>({
    isOpen: false,
    version: null,
  });

  const queryClient = useQueryClient();

  const { data: versions, isLoading } = useQuery<Version[]>(
    ['versions', identityId],
    () => apiClient.getVersions(identityId),
    { enabled: !!identityId }
  );

  const createVersionMutation = useMutation(() => apiClient.createVersion(identityId), {
    onSuccess: () => {
      queryClient.invalidateQueries(['versions', identityId]);
    },
  });

  const restoreMutation = useMutation(
    (version: string) => apiClient.restoreVersion(identityId, version),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['versions', identityId]);
        setRestoreConfirm({ isOpen: false, version: null });
        router.push(`/identities/${identityId}`);
      },
    }
  );

  const handleCreateVersion = (): void => {
    createVersionMutation.mutate();
  };

  const handleRestore = (version: string): void => {
    setRestoreConfirm({ isOpen: true, version });
  };

  const handleConfirmRestore = (): void => {
    if (restoreConfirm.version) {
      restoreMutation.mutate(restoreConfirm.version);
    }
  };

  const handleCancelRestore = (): void => {
    setRestoreConfirm({ isOpen: false, version: null });
  };

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Versions</h1>
          <Button onClick={handleCreateVersion} isLoading={createVersionMutation.isLoading}>
            Create Version
          </Button>
        </div>

        {!versions || versions.length === 0 ? (
          <Card>
            <EmptyState
              title="No versions"
              description="Create a version to save the current state of this identity"
              actionLabel="Create Version"
              onAction={handleCreateVersion}
            />
          </Card>
        ) : (
          <div className="space-y-4">
            {versions.map((version) => (
              <Card key={version.version_id}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">Version {version.version_id}</h3>
                    {version.description && (
                      <p className="text-sm text-gray-600">{version.description}</p>
                    )}
                    <p className="text-xs text-gray-500">{formatDate(version.created_at)}</p>
                  </div>
                  <Button variant="secondary" onClick={() => handleRestore(version.version_id)}>
                    Restore
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}

        <ConfirmDialog
          isOpen={restoreConfirm.isOpen}
          onClose={handleCancelRestore}
          onConfirm={handleConfirmRestore}
          title="Restore Version"
          message="Are you sure you want to restore this version? Current changes will be lost."
          confirmLabel="Restore"
          cancelLabel="Cancel"
          variant="danger"
          isLoading={restoreMutation.isLoading}
        />
      </div>
    </PageLayout>
  );
};

export default VersionsPage;



