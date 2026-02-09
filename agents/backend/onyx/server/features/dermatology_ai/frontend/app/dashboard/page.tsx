'use client';

import React from 'react';
import { apiClient } from '@/lib/api/client';
import { DashboardOverview, UserAnalytics } from '@/lib/types/api';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { StatsCard } from '@/components/dashboard/StatsCard';
import { ProgressChart } from '@/components/dashboard/ProgressChart';
import { SkeletonCard } from '@/components/ui/Skeleton';
import { BarChart3, TrendingUp, AlertCircle, Clock, Activity, Target } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import Link from 'next/link';
import { useAuth } from '@/lib/contexts/AuthContext';
import { ProtectedPage } from '@/components/auth/ProtectedPage';
import { useAsyncData } from '@/lib/hooks/useAsyncData';
import { PageLayout } from '@/components/layout/PageLayout';
import { PageHeader } from '@/components/layout/PageHeader';
import { Grid } from '@/components/ui/Grid';
import { SkeletonGrid } from '@/components/ui/SkeletonGrid';

interface DashboardData {
  overview: DashboardOverview;
  analytics: UserAnalytics;
}

export default function DashboardPage() {
  const { user } = useAuth();

  const { data, isLoading } = useAsyncData<DashboardData>({
    fetchFn: async () => {
      if (!user) throw new Error('User not authenticated');
      const [overviewData, analyticsData] = await Promise.all([
        apiClient.getDashboardOverview(),
        apiClient.getUserAnalytics(user.id),
      ]);
      return {
        overview: overviewData,
        analytics: analyticsData as UserAnalytics,
      };
    },
    enabled: !!user,
    errorMessage: 'Failed to load dashboard. Please try again.',
  });

  const overview = data?.overview || null;
  const analytics = data?.analytics || null;

  return (
    <ProtectedPage message="Sign in to access your dashboard">
      <PageLayout>
        <PageHeader
          title="Dashboard"
          description="Your skin health at a glance"
        />

        {/* Stats Grid */}
        {isLoading ? (
          <SkeletonGrid count={4} cols={{ base: 1, md: 2, lg: 4 }} gap={6} className="mb-8" />
        ) : (
          <Grid cols={{ base: 1, md: 2, lg: 4 }} gap={6} className="mb-8">
            <StatsCard
              title="Total Analyses"
              value={overview?.total_analyses || 0}
              icon={BarChart3}
              trend={{ value: 12, isPositive: true }}
              iconColor="text-primary-600 dark:text-primary-400"
            />
            <StatsCard
              title="Average Score"
              value={overview?.average_score?.toFixed(1) || '0.0'}
              icon={TrendingUp}
              trend={{ value: 5, isPositive: true }}
              iconColor="text-green-600 dark:text-green-400"
            />
            <StatsCard
              title="Active Alerts"
              value={overview?.active_alerts || 0}
              icon={AlertCircle}
              iconColor="text-yellow-600 dark:text-yellow-400"
            />
            <StatsCard
              title="Last Analysis"
              value={
                overview?.recent_analyses?.[0]?.timestamp
                  ? new Date(overview.recent_analyses[0].timestamp).toLocaleDateString('en-US', {
                      day: 'numeric',
                      month: 'short',
                    })
                  : 'N/A'
              }
              icon={Clock}
              subtitle="Last analysis date"
              iconColor="text-blue-600 dark:text-blue-400"
            />
          </Grid>
        )}

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {isLoading ? (
            <>
              <SkeletonCard />
              <SkeletonCard />
            </>
          ) : (
            <>
              {overview?.trends && overview.trends.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Progress Over Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ProgressChart
                      data={overview.trends[0]?.values || []}
                      type="area"
                      color="#0ea5e9"
                    />
                  </CardContent>
                </Card>
              )}

              {analytics && analytics.average_scores && (
                <Card>
                  <CardHeader>
                    <CardTitle>Quality Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={[
                        { metric: 'Texture', value: analytics.average_scores.texture_score },
                        { metric: 'Hydration', value: analytics.average_scores.hydration_score },
                        { metric: 'Elasticity', value: analytics.average_scores.elasticity_score },
                        { metric: 'Pigmentation', value: analytics.average_scores.pigmentation_score },
                        { metric: 'Pores', value: analytics.average_scores.pore_size_score },
                        { metric: 'Wrinkles', value: analytics.average_scores.wrinkles_score },
                      ]}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} />
                        <Radar
                          name="Score"
                          dataKey="value"
                          stroke="#0ea5e9"
                          fill="#0ea5e9"
                          fillOpacity={0.6}
                        />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>

          {analytics && (
            <Card>
              <CardHeader>
                <CardTitle>Most Common Conditions</CardTitle>
              </CardHeader>
              <CardContent>
                {analytics.most_common_conditions && analytics.most_common_conditions.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={analytics.most_common_conditions}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="condition" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#0ea5e9" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-gray-500 text-center py-8">
                    No condition data available yet
                  </p>
                )}
              </CardContent>
            </Card>
          )}

        {/* Recent Analyses */}
        {overview?.recent_analyses && overview.recent_analyses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Recent Analyses</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {overview.recent_analyses.slice(0, 5).map((analysis) => (
                  <div
                    key={analysis.record_id}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div>
                      <p className="font-medium text-gray-900">
                        Analysis #{analysis.record_id.slice(0, 8)}
                      </p>
                      <p className="text-sm text-gray-600">
                        {new Date(analysis.timestamp).toLocaleString('en-US')}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-bold text-primary-600">
                        {analysis.analysis.quality_scores.overall_score.toFixed(1)}
                      </p>
                      <p className="text-xs text-gray-500">Score</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 text-center">
                <Link
                  href="/history"
                  className="text-primary-600 hover:text-primary-700 font-medium"
                >
                  View full history →
                </Link>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Quick Actions */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link
                href="/"
                className="p-4 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors text-center"
              >
                <p className="font-medium text-primary-900">New Analysis</p>
                <p className="text-sm text-primary-700 mt-1">
                  New analysis
                </p>
              </Link>
              <Link
                href="/history"
                className="p-4 bg-secondary-50 rounded-lg hover:bg-secondary-100 transition-colors text-center"
              >
                <p className="font-medium text-secondary-900">View History</p>
                <p className="text-sm text-secondary-700 mt-1">
                  View history
                </p>
              </Link>
              <Link
                href="/products"
                className="p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors text-center"
              >
                <p className="font-medium text-green-900">Products</p>
                <p className="text-sm text-green-700 mt-1">
                  Explore products
                </p>
              </Link>
            </div>
          </CardContent>
        </Card>
      </PageLayout>
    </ProtectedPage>
  );
}

