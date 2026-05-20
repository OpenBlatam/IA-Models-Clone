'use client';

import React, { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { HistoryRecord } from '@/lib/types/api';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { SearchBar } from '@/components/ui/SearchBar';
import { FilterBar, FilterOption } from '@/components/ui/FilterBar';
import { History, Calendar, TrendingUp, TrendingDown } from 'lucide-react';
import { formatHistoryDate } from '@/lib/utils/dateUtils';
import { useAuth } from '@/lib/contexts/AuthContext';
import { ProtectedPage } from '@/components/auth/ProtectedPage';
import { EmptyState } from '@/lib/utils/emptyStates';
import { useAsyncData } from '@/lib/hooks/useAsyncData';
import { useFilter } from '@/lib/hooks/useFilter';
import { toastMessages, showInfo } from '@/lib/utils/toastUtils';
import { PageLayout } from '@/components/layout/PageLayout';
import { PageHeader } from '@/components/layout/PageHeader';
import { getScoreColor, getScoreBgColor } from '@/lib/utils/scoreUtils';

export default function HistoryPage() {
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null);
  const { user } = useAuth();

  const { data, isLoading } = useAsyncData<HistoryRecord[]>({
    fetchFn: async () => {
      if (!user) throw new Error('User not authenticated');
      const response = await apiClient.getHistory(user.id);
      return response.records || [];
    },
    enabled: !!user,
    errorMessage: 'Failed to load history',
  });

  const records = data || [];
  const [searchQuery, setSearchQuery] = useState('');
  const [skinTypeFilter, setSkinTypeFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');

  const {
    filteredItems: filteredRecords,
  } = useFilter<HistoryRecord>({
    items: records,
    filterFn: (record) => {
      // Search filter
      if (searchQuery) {
        const searchLower = searchQuery.toLowerCase();
        const matchesSearch =
          record.analysis.skin_type.toLowerCase().includes(searchLower) ||
          record.notes?.toLowerCase().includes(searchLower) ||
          record.analysis.conditions?.some((c) =>
            c.name.toLowerCase().includes(searchLower)
          );
        if (!matchesSearch) return false;
      }

      // Skin type filter
      if (skinTypeFilter && record.analysis.skin_type !== skinTypeFilter) {
        return false;
      }

      // Date filter
      if (dateFilter) {
        const recordDate = new Date(record.timestamp);
        const now = new Date();
        switch (dateFilter) {
          case 'today':
            if (recordDate.toDateString() !== now.toDateString()) return false;
            break;
          case 'week':
            const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
            if (recordDate < weekAgo) return false;
            break;
          case 'month':
            const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
            if (recordDate < monthAgo) return false;
            break;
          default:
            break;
        }
      }

      return true;
    },
    dependencies: [searchQuery, skinTypeFilter, dateFilter],
  });

  const skinTypeOptions: FilterOption[] = [
    { label: 'Dry', value: 'dry' },
    { label: 'Oily', value: 'oily' },
    { label: 'Combination', value: 'combination' },
    { label: 'Normal', value: 'normal' },
    { label: 'Sensitive', value: 'sensitive' },
  ];

  const dateOptions: FilterOption[] = [
    { label: 'Today', value: 'today' },
    { label: 'This week', value: 'week' },
    { label: 'This month', value: 'month' },
  ];


  return (
    <ProtectedPage>
      <PageLayout>
        <PageHeader
          title="Analysis History"
          description="Review your analysis history and track progress"
          icon={History}
        />

        {/* Search and Filters */}
        <div className="mb-6 space-y-4">
          <SearchBar
            placeholder="Search by skin type, conditions, or notes..."
            onSearch={setSearchQuery}
            className="w-full"
          />
          <FilterBar
            filters={[
              {
                label: 'Skin Type',
                options: skinTypeOptions,
                value: skinTypeFilter,
                onChange: setSkinTypeFilter,
              },
              {
                label: 'Date',
                options: dateOptions,
                value: dateFilter,
                onChange: setDateFilter,
              },
            ]}
            onClearAll={() => {
              setSkinTypeFilter('');
              setDateFilter('');
              setSearchQuery('');
            }}
          />
        </div>

        {filteredRecords.length === 0 ? (
          <EmptyState
            icon={<History className="h-16 w-16 text-gray-400" />}
            title="No analyses yet"
            description="Start by analyzing an image to build your history"
            actionLabel="Start your first analysis"
            actionHref="/"
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Records List */}
            <div className="lg:col-span-2 space-y-4">
              {filteredRecords.length > 0 && (
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  Showing {filteredRecords.length} of {records.length} analyses
                </p>
              )}
              {filteredRecords.map((record) => (
                <Card
                  key={record.record_id}
                  className={`cursor-pointer transition-all ${
                    selectedRecord?.record_id === record.record_id
                      ? 'ring-2 ring-primary-500'
                      : 'hover:shadow-lg'
                  }`}
                  onClick={() => setSelectedRecord(record)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <Calendar className="h-4 w-4 text-gray-400" />
                          <span className="text-sm text-gray-600">
                            {formatHistoryDate(record.timestamp)}
                          </span>
                        </div>
                        <div className="mb-3">
                          <span className="text-xs font-medium text-gray-500 uppercase">
                            Skin Type
                          </span>
                          <p className="text-lg font-semibold text-gray-900 capitalize">
                            {record.analysis.skin_type}
                          </p>
                        </div>
                        {record.analysis.conditions && record.analysis.conditions.length > 0 && (
                          <div className="mb-3">
                            <span className="text-xs font-medium text-gray-500 uppercase">
                              Detected Conditions
                            </span>
                            <div className="flex flex-wrap gap-2 mt-1">
                              {record.analysis.conditions.slice(0, 3).map((condition, idx) => (
                                <span
                                  key={idx}
                                  className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                                >
                                  {condition.name}
                                </span>
                              ))}
                              {record.analysis.conditions.length > 3 && (
                                <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                                  +{record.analysis.conditions.length - 3} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                        {record.notes && (
                          <p className="text-sm text-gray-600 mt-2 italic">
                            "{record.notes}"
                          </p>
                        )}
                      </div>
                      <div className="ml-4 text-right">
                        <div
                          className={`inline-flex items-center justify-center w-20 h-20 rounded-full text-2xl font-bold ${getScoreColor(
                            record.analysis.quality_scores.overall_score
                          )} ${getScoreBgColor(
                            record.analysis.quality_scores.overall_score
                          )}`}
                        >
                          {record.analysis.quality_scores.overall_score.toFixed(0)}
                        </div>
                        <p className="text-xs text-gray-500 mt-2">Score</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Record Details */}
            <div className="lg:col-span-1">
              {selectedRecord ? (
                <Card className="sticky top-8">
                  <CardHeader>
                    <CardTitle>Analysis Details</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Date</p>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {formatHistoryDate(selectedRecord.timestamp)}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Quality Metrics</p>
                      <div className="space-y-2">
                        {Object.entries(selectedRecord.analysis.quality_scores)
                          .filter(([key]) => key !== 'overall_score')
                          .map(([key, value]) => (
                            <div key={key} className="flex items-center justify-between">
                              <span className="text-sm text-gray-700 capitalize">
                                {key.replace('_score', '').replace('_', ' ')}
                              </span>
                              <div className="flex items-center space-x-2">
                                <div className="w-24 bg-gray-200 rounded-full h-2">
                                  <div
                                    className={`h-2 rounded-full ${
                                      value >= 80
                                        ? 'bg-green-500'
                                        : value >= 60
                                        ? 'bg-yellow-500'
                                        : 'bg-red-500'
                                    }`}
                                    style={{ width: `${value}%` }}
                                  />
                                </div>
                                <span className="text-sm font-medium w-12 text-right">
                                  {value.toFixed(1)}
                                </span>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>

                    {selectedRecord.analysis.conditions &&
                      selectedRecord.analysis.conditions.length > 0 && (
                        <div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Conditions</p>
                          <div className="space-y-2">
                            {selectedRecord.analysis.conditions.map((condition, idx) => (
                              <div
                                key={idx}
                                className="p-2 bg-gray-50 rounded text-sm"
                              >
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-medium capitalize">
                                    {condition.name}
                                  </span>
                                  <span className="text-xs text-gray-500">
                                    {(condition.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <span
                                  className={`text-xs px-2 py-1 rounded ${
                                    condition.severity === 'severe'
                                      ? 'bg-red-100 text-red-800'
                                      : condition.severity === 'moderate'
                                      ? 'bg-yellow-100 text-yellow-800'
                                      : 'bg-blue-100 text-blue-800'
                                  }`}
                                >
                                  {condition.severity}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                    <div className="pt-4 border-t">
                      <Button
                        variant="outline"
                        className="w-full"
                        onClick={() => {
                          // TODO: Implement comparison
                          showInfo(toastMessages.comingSoon);
                        }}
                      >
                        Compare with Another
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className="p-8 text-center">
                  <p className="text-gray-600 dark:text-gray-400">Select an analysis to view details</p>
                </Card>
              )}
            </div>
          </div>
        )}
      </PageLayout>
    </ProtectedPage>
  );
}

