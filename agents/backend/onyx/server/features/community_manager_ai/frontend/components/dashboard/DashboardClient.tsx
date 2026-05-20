/**
 * Dashboard Client Component
 * Client-side wrapper for dashboard with state management
 */

'use client';

import { useState, Suspense } from 'react';
import { Layout } from '@/components/layout/Layout';
import { PageHeader } from '@/components/ui/PageHeader';
import { Loading } from '@/components/ui/Loading';
import { Alert } from '@/components/ui/Alert';
import { DashboardStats } from './DashboardStats';
import { DashboardCharts } from './DashboardCharts';
import { useDashboardOverview, useDashboardEngagement } from '@/hooks/useDashboard';
import { useTranslations } from 'next-intl';
import { Select } from '@/components/ui/Select';

/**
 * Dashboard client component with data fetching
 */
export const DashboardClient = () => {
  const [days, setDays] = useState(7);
  const t = useTranslations('dashboard');
  const tCommon = useTranslations('common');

  const { data: overview, isLoading: overviewLoading, error: overviewError } = useDashboardOverview(days);
  const { data: engagement, isLoading: engagementLoading, error: engagementError } = useDashboardEngagement(days);

  const loading = overviewLoading || engagementLoading;
  const error = overviewError || engagementError;

  return (
    <Layout>
      <div className="space-y-6">
        <PageHeader
          title={t('title')}
          description={t('subtitle')}
          actions={
            <Select
              value={days.toString()}
              onChange={(e) => setDays(Number(e.target.value))}
              options={[
                { value: '7', label: t('last7Days') },
                { value: '30', label: t('last30Days') },
                { value: '90', label: t('last90Days') },
              ]}
              className="w-[180px]"
              aria-label="Período de tiempo"
            />
          }
        />

        {error ? (
          <Alert variant="error" title={tCommon('error')}>
            {error instanceof Error ? error.message : tCommon('error')}
          </Alert>
        ) : (
          <>
            <Suspense fallback={<div className="flex items-center justify-center h-64"><Loading size="lg" text={tCommon('loading')} /></div>}>
              <DashboardStats overview={overview} isLoading={loading} t={t} />
            </Suspense>

            <Suspense fallback={<div className="flex items-center justify-center h-64"><Loading size="lg" text={tCommon('loading')} /></div>}>
              <DashboardCharts engagement={engagement} overview={overview} isLoading={loading} t={t} />
            </Suspense>
          </>
        )}
      </div>
    </Layout>
  );
};

