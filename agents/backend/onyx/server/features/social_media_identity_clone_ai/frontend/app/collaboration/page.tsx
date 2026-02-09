'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { apiClient } from '@/lib/api/client';
import type { IdentityProfile } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import ConfirmDialog from '@/components/UI/ConfirmDialog';
import { PermissionLevel } from '@/types';

const CollaborationPage = (): JSX.Element => {
  const queryClient = useQueryClient();
  const [showShareForm, setShowShareForm] = useState(false);
  const [shareFormData, setShareFormData] = useState({
    identity_id: '',
    shared_with_user_id: '',
    permission_level: PermissionLevel.VIEWER,
    shared_by_user_id: '',
  });
  const [revokeConfirm, setRevokeConfirm] = useState<{ isOpen: boolean; shareId: string | null }>({
    isOpen: false,
    shareId: null,
  });

  const { data: sharedIdentities, isLoading } = useQuery<IdentityProfile[]>('shared-identities', () =>
    apiClient.getSharedIdentities()
  );

  const shareMutation = useMutation(
    () =>
      apiClient.shareIdentity(
        shareFormData.identity_id,
        shareFormData.shared_with_user_id,
        shareFormData.permission_level,
        shareFormData.shared_by_user_id
      ),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('shared-identities');
        setShowShareForm(false);
        setShareFormData({
          identity_id: '',
          shared_with_user_id: '',
          permission_level: PermissionLevel.VIEWER,
          shared_by_user_id: '',
        });
      },
    }
  );

  const revokeMutation = useMutation((shareId: string) => apiClient.revokeShare(shareId), {
    onSuccess: () => {
      queryClient.invalidateQueries('shared-identities');
      setRevokeConfirm({ isOpen: false, shareId: null });
    },
  });

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    shareMutation.mutate();
  };

  const handleIdentityIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setShareFormData({ ...shareFormData, identity_id: e.target.value });
  };

  const handleUserIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setShareFormData({ ...shareFormData, shared_with_user_id: e.target.value });
  };

  const handlePermissionChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setShareFormData({ ...shareFormData, permission_level: e.target.value as PermissionLevel });
  };

  const handleOwnerIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setShareFormData({ ...shareFormData, shared_by_user_id: e.target.value });
  };

  const handleToggleForm = (): void => {
    setShowShareForm(!showShareForm);
  };

  const handleRevoke = (shareId: string): void => {
    setRevokeConfirm({ isOpen: true, shareId });
  };

  const handleConfirmRevoke = (): void => {
    if (revokeConfirm.shareId) {
      revokeMutation.mutate(revokeConfirm.shareId);
    }
  };

  const handleCancelRevoke = (): void => {
    setRevokeConfirm({ isOpen: false, shareId: null });
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
          <h1 className="text-3xl font-bold">Collaboration</h1>
          <Button onClick={handleToggleForm}>
            {showShareForm ? 'Cancel' : 'Share Identity'}
          </Button>
        </div>

        {showShareForm && (
          <Card title="Share Identity" className="mb-8">
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <Input
                  label="Identity ID"
                  value={shareFormData.identity_id}
                  onChange={handleIdentityIdChange}
                  required
                />
                <Input
                  label="Share With User ID"
                  value={shareFormData.shared_with_user_id}
                  onChange={handleUserIdChange}
                  required
                />
                <Input
                  label="Shared By User ID"
                  value={shareFormData.shared_by_user_id}
                  onChange={handleOwnerIdChange}
                  required
                />
                <Select
                  label="Permission Level"
                  value={shareFormData.permission_level}
                  onChange={handlePermissionChange}
                  options={[
                    { value: PermissionLevel.VIEWER, label: 'Viewer' },
                    { value: PermissionLevel.EDITOR, label: 'Editor' },
                    { value: PermissionLevel.ADMIN, label: 'Admin' },
                  ]}
                />
                <Button type="submit" isLoading={shareMutation.isLoading} className="w-full">
                  Share
                </Button>
              </div>
            </form>
          </Card>
        )}

        {!sharedIdentities || sharedIdentities.length === 0 ? (
          <Card>
            <EmptyState
              title="No shared identities"
              description="Share an identity to collaborate with others"
              actionLabel="Share Identity"
              onAction={handleToggleForm}
            />
          </Card>
        ) : (
          <div className="space-y-4">
            {sharedIdentities.map((identity) => (
              <Card key={identity.profile_id}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">{identity.username}</h3>
                    {identity.display_name && (
                      <p className="text-sm text-gray-600">{identity.display_name}</p>
                    )}
                  </div>
                  <Button
                    variant="danger"
                    onClick={() => handleRevoke(identity.profile_id)}
                    className="text-sm"
                  >
                    Revoke
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}

        <ConfirmDialog
          isOpen={revokeConfirm.isOpen}
          onClose={handleCancelRevoke}
          onConfirm={handleConfirmRevoke}
          title="Revoke Share"
          message="Are you sure you want to revoke access to this identity?"
          confirmLabel="Revoke"
          cancelLabel="Cancel"
          variant="danger"
          isLoading={revokeMutation.isLoading}
        />
      </div>
    </PageLayout>
  );
};

export default CollaborationPage;



