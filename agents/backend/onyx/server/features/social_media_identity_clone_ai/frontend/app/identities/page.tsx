'use client';

import { useRouter } from 'next/navigation';
import { useIdentities } from '@/lib/modules/api';
import PageHeader from '@/components/Layout/PageHeader';
import IdentityCard from '@/components/features/IdentityCard';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';

const IdentitiesPage = (): JSX.Element => {
  const router = useRouter();
  
  const { data: identities, isLoading } = useIdentities();


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
        <PageHeader
          title="Identities"
          action={{
            label: 'Build New Identity',
            onClick: () => router.push('/build-identity'),
          }}
        />

        {!identities || identities.length === 0 ? (
          <Card>
            <EmptyState
              title="No identities found"
              description="Get started by creating your first identity profile"
              actionLabel="Create Your First Identity"
              actionHref="/build-identity"
            />
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {identities.map((identity) => (
              <IdentityCard key={identity.profile_id} identity={identity} />
            ))}
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default IdentitiesPage;

