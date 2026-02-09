'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import ConfirmDialog from '@/components/UI/ConfirmDialog';
import { formatDate } from '@/lib/utils';

interface Backup {
  backup_id: string;
  created_at: string;
  size?: number;
}

const BackupsPage = (): JSX.Element => {
  const queryClient = useQueryClient();
  const [restoreConfirm, setRestoreConfirm] = useState<{ isOpen: boolean; backupId: string | null }>({
    isOpen: false,
    backupId: null,
  });

  const { data: backups, isLoading } = useQuery<Backup[]>('backups', () => apiClient.listBackups());

  const createMutation = useMutation(() => apiClient.createBackup(), {
    onSuccess: () => {
      queryClient.invalidateQueries('backups');
    },
  });

  const restoreMutation = useMutation((backupId: string) => apiClient.restoreBackup(backupId), {
    onSuccess: () => {
      queryClient.invalidateQueries('backups');
      setRestoreConfirm({ isOpen: false, backupId: null });
    },
  });

  const cleanupMutation = useMutation(() => apiClient.cleanupBackups(), {
    onSuccess: () => {
      queryClient.invalidateQueries('backups');
    },
  });

  const handleCreateBackup = (): void => {
    createMutation.mutate();
  };

  const handleRestore = (backupId: string): void => {
    setRestoreConfirm({ isOpen: true, backupId });
  };

  const handleConfirmRestore = (): void => {
    if (restoreConfirm.backupId) {
      restoreMutation.mutate(restoreConfirm.backupId);
    }
  };

  const handleCancelRestore = (): void => {
    setRestoreConfirm({ isOpen: false, backupId: null });
  };

  const handleCleanup = (): void => {
    cleanupMutation.mutate();
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
          <h1 className="text-3xl font-bold">Backups</h1>
          <div className="flex gap-2">
            <Button onClick={handleCreateBackup} isLoading={createMutation.isLoading}>
              Create Backup
            </Button>
            <Button variant="secondary" onClick={handleCleanup} isLoading={cleanupMutation.isLoading}>
              Cleanup
            </Button>
          </div>
        </div>

        {!backups || backups.length === 0 ? (
          <Card>
            <EmptyState
              title="No backups"
              description="Create a backup to protect your data"
              actionLabel="Create Backup"
              onAction={handleCreateBackup}
            />
          </Card>
        ) : (
          <div className="space-y-4">
            {backups.map((backup) => (
              <Card key={backup.backup_id}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">Backup {backup.backup_id}</h3>
                    <p className="text-sm text-gray-600">{formatDate(backup.created_at)}</p>
                    {backup.size && (
                      <p className="text-xs text-gray-500">Size: {(backup.size / 1024).toFixed(2)} KB</p>
                    )}
                  </div>
                  <Button variant="secondary" onClick={() => handleRestore(backup.backup_id)}>
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
          title="Restore Backup"
          message="Are you sure you want to restore this backup? This will overwrite current data."
          confirmLabel="Restore"
          cancelLabel="Cancel"
          variant="danger"
          isLoading={restoreMutation.isLoading}
        />
      </div>
    </PageLayout>
  );
};

export default BackupsPage;

