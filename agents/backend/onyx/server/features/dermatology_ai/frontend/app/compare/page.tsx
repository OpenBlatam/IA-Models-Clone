'use client';

import React, { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { HistoryRecord, ComparisonResponse } from '@/lib/types/api';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Loading } from '@/components/ui/Loading';
import { Compare as CompareIcon } from 'lucide-react';
import { toastMessages, showError, showSuccess } from '@/lib/utils/toastUtils';
import { useAuth } from '@/lib/contexts/AuthContext';
import { ProtectedPage } from '@/components/auth/ProtectedPage';
import { PageLayout } from '@/components/layout/PageLayout';
import { PageHeader } from '@/components/layout/PageHeader';
import { useAsyncData } from '@/lib/hooks/useAsyncData';
import { useMutation } from '@/lib/hooks/useMutation';
import { Grid } from '@/components/ui/Grid';
import { Select } from '@/components/ui/Select';
import { MetricBox } from '@/components/ui/MetricBox';
import { StatItem } from '@/components/ui/StatItem';
import { formatHistoryDate } from '@/lib/utils/dateUtils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function ComparePage() {
  const { user } = useAuth();
  const [selectedRecord1, setSelectedRecord1] = useState<string | null>(null);
  const [selectedRecord2, setSelectedRecord2] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);

  const { data: records, isLoading } = useAsyncData<HistoryRecord[]>({
    fetchFn: async () => {
      if (!user) throw new Error('User not authenticated');
      const response = await apiClient.getHistory(user.id);
      return response.records || [];
    },
    enabled: !!user,
    errorMessage: toastMessages.loadFailed,
  });

  const compareMutation = useMutation<ComparisonResponse, { record1: string; record2: string }>({
    mutationFn: async ({ record1, record2 }) => {
      if (!record1 || !record2) {
        throw new Error(toastMessages.selectTwoAnalyses);
      }
      if (record1 === record2) {
        throw new Error(toastMessages.selectDifferentAnalyses);
      }
      return await apiClient.compareHistory(record1, record2);
    },
    onSuccess: (result) => {
      setComparison(result);
    },
    successMessage: toastMessages.comparisonComplete,
    errorMessage: toastMessages.comparisonFailed,
  });

  const handleCompare = () => {
    if (!selectedRecord1 || !selectedRecord2) return;
    compareMutation.mutate({ record1: selectedRecord1, record2: selectedRecord2 });
  };

  return (
    <ProtectedPage message="Sign in to compare analyses">
      <PageLayout>
        <PageHeader
          title="Compare Analyses"
          description="Compare analyses to track your progress"
          icon={CompareIcon}
        />

        {isLoading ? (
          <Loading fullScreen text="Loading history..." />
        ) : (
          <div className="space-y-6">
            {/* Selection */}
            <Grid cols={{ base: 1, md: 2 }} gap={6}>
              <Card>
                <CardHeader>
                  <CardTitle>Analysis 1</CardTitle>
                </CardHeader>
                <CardContent>
                  <Select
                    value={selectedRecord1 || ''}
                    onChange={(e) => setSelectedRecord1(e.target.value)}
                    options={records.map((record) => ({
                      value: record.record_id,
                      label: `${formatHistoryDate(record.timestamp)} - Score: ${record.analysis.quality_scores.overall_score.toFixed(1)}`,
                    }))}
                    placeholder="Select an analysis"
                    fullWidth
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Analysis 2</CardTitle>
                </CardHeader>
                <CardContent>
                  <Select
                    value={selectedRecord2 || ''}
                    onChange={(e) => setSelectedRecord2(e.target.value)}
                    options={records.map((record) => ({
                      value: record.record_id,
                      label: `${formatHistoryDate(record.timestamp)} - Score: ${record.analysis.quality_scores.overall_score.toFixed(1)}`,
                    }))}
                    placeholder="Select an analysis"
                    fullWidth
                  />
                </CardContent>
              </Card>
            </div>

            <div className="text-center">
              <Button
                onClick={handleCompare}
                isLoading={compareMutation.isLoading}
                disabled={!selectedRecord1 || !selectedRecord2}
                size="lg"
              >
                Compare Analyses
              </Button>
            </div>

            {/* Comparison Results */}
            {comparison && (
              <div className="space-y-6 animate-fade-in">
                <Card>
                  <CardHeader>
                    <CardTitle>Comparison Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Grid cols={{ base: 1, md: 3 }} gap={4}>
                      <MetricBox
                        label="Analysis 1"
                        value={comparison.comparison.record1.analysis.quality_scores.overall_score.toFixed(1)}
                        subtitle={formatHistoryDate(comparison.comparison.record1.timestamp)}
                        variant="primary"
                      />
                      <MetricBox
                        label="Overall Improvement"
                        value={
                          comparison.comparison.overall_improvement > 0
                            ? `+${comparison.comparison.overall_improvement.toFixed(1)}`
                            : comparison.comparison.overall_improvement.toFixed(1)
                        }
                        variant={
                          comparison.comparison.overall_improvement > 0
                            ? 'success'
                            : comparison.comparison.overall_improvement < 0
                            ? 'danger'
                            : 'default'
                        }
                      />
                      <MetricBox
                        label="Analysis 2"
                        value={comparison.comparison.record2.analysis.quality_scores.overall_score.toFixed(1)}
                        subtitle={formatHistoryDate(comparison.comparison.record2.timestamp)}
                        variant="primary"
                      />
                    </Grid>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Changes by Metric</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {comparison.comparison.differences.map((diff, index) => (
                        <StatItem
                          key={index}
                          label={diff.metric.replace('_', ' ')}
                          value={`${diff.change > 0 ? '+' : ''}${diff.change.toFixed(1)} (${diff.percentage > 0 ? '+' : ''}${diff.percentage.toFixed(1)}%)`}
                          icon={diff.change > 0 ? TrendingUp : diff.change < 0 ? TrendingDown : Minus}
                          iconColor={
                            diff.change > 0
                              ? 'text-green-600 dark:text-green-400'
                              : diff.change < 0
                              ? 'text-red-600 dark:text-red-400'
                              : 'text-gray-600 dark:text-gray-400'
                          }
                        />
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}
      </PageLayout>
    </ProtectedPage>
  );
}

