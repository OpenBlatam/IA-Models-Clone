/**
 * Dashboard Charts Component
 * Client Component for displaying dashboard charts with dynamic import
 */

'use client';

import dynamic from 'next/dynamic';
import { Suspense } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { TrendingUp } from 'lucide-react';
import type { EngagementSummary } from '@/types';

// Dynamic imports for charts to reduce initial bundle size
const LineChart = dynamic(
  () => import('recharts').then((mod) => mod.LineChart),
  { ssr: false, loading: () => <Skeleton className="h-[300px] w-full" /> }
);

const BarChart = dynamic(
  () => import('recharts').then((mod) => mod.BarChart),
  { ssr: false, loading: () => <Skeleton className="h-[300px] w-full" /> }
);

const ResponsiveContainer = dynamic(
  () => import('recharts').then((mod) => mod.ResponsiveContainer),
  { ssr: false }
);

const Line = dynamic(
  () => import('recharts').then((mod) => mod.Line),
  { ssr: false }
);

const Bar = dynamic(
  () => import('recharts').then((mod) => mod.Bar),
  { ssr: false }
);

const XAxis = dynamic(
  () => import('recharts').then((mod) => mod.XAxis),
  { ssr: false }
);

const YAxis = dynamic(
  () => import('recharts').then((mod) => mod.YAxis),
  { ssr: false }
);

const CartesianGrid = dynamic(
  () => import('recharts').then((mod) => mod.CartesianGrid),
  { ssr: false }
);

const Tooltip = dynamic(
  () => import('recharts').then((mod) => mod.Tooltip),
  { ssr: false }
);

interface DashboardChartsProps {
  engagement: EngagementSummary | undefined;
  overview: { average_engagement_rate?: number } | undefined;
  isLoading: boolean;
  t: (key: string) => string;
}

/**
 * Dashboard charts component with lazy loading
 */
export const DashboardCharts = ({ engagement, overview, isLoading, t }: DashboardChartsProps) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[300px] w-full" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[300px] w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  const engagementData =
    engagement?.trends?.map((item) => ({
      date: new Date(item.date).toLocaleDateString('es-ES', { month: 'short', day: 'numeric' }),
      engagement: item.engagement,
    })) || [];

  const platformData = engagement?.engagement_by_platform
    ? Object.entries(engagement.engagement_by_platform).map(([platform, value]) => ({
        platform,
        engagement: value,
      }))
    : [];

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>{t('engagementRate')}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-5 w-5 text-green-600 dark:text-green-400" />
            <span className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {overview?.average_engagement_rate?.toFixed(2) || 0}%
            </span>
          </div>
          {engagementData.length > 0 ? (
            <Suspense fallback={<Skeleton className="h-[300px] w-full" />}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={engagementData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" className="dark:stroke-gray-700" />
                  <XAxis dataKey="date" stroke="#6b7280" className="dark:stroke-gray-400" />
                  <YAxis stroke="#6b7280" className="dark:stroke-gray-400" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                  />
                  <Line type="monotone" dataKey="engagement" stroke="#0ea5e9" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Suspense>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
              {t('noData')}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t('engagementByPlatform')}</CardTitle>
        </CardHeader>
        <CardContent>
          {platformData.length > 0 ? (
            <Suspense fallback={<Skeleton className="h-[300px] w-full" />}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={platformData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="platform" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="engagement" fill="#0ea5e9" />
                </BarChart>
              </ResponsiveContainer>
            </Suspense>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
              {t('noData')}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};


