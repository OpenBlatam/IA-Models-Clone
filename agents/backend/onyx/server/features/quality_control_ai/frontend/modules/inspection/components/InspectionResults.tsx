'use client';

import { memo, useMemo } from 'react';
import { useQualityControlStore } from '@/lib/store';
import { formatDate, getStatusLabel } from '@/lib/utils';
import Card from '@/components/ui/Card';
import EmptyState from '@/components/ui/EmptyState';
import StatCard from '@/components/ui/StatCard';
import QualityScore from './QualityScore';
import StatusBadge from './StatusBadge';
import SeverityCounts from './SeverityCounts';
import DefectList from './DefectList';

const InspectionResults = memo((): JSX.Element => {
  const { currentResult } = useQualityControlStore();

  const summaryCard = useMemo(() => {
    if (!currentResult) return null;

    const { summary, quality_score, defects, objects } = currentResult;

    return (
      <>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <QualityScore score={quality_score} />
          <StatCard
            label="Status"
            value={getStatusLabel(summary.status)}
            valueClassName="text-2xl font-semibold"
          />
          <StatCard
            label="Defects"
            value={defects.length}
            valueClassName="text-3xl font-bold"
          />
          <StatCard
            label="Objects"
            value={objects.length}
            valueClassName="text-3xl font-bold"
          />
        </div>

        <div className="mb-6">
          <StatusBadge status={summary.status} recommendation={summary.recommendation} />
        </div>

        <SeverityCounts counts={summary.severity_counts} />
      </>
    );
  }, [currentResult]);

  if (!currentResult) {
    return (
      <Card title="Inspection Results">
        <EmptyState
          title="No inspection results yet"
          description="Start an inspection to see results here"
        />
      </Card>
    );
  }

  return (
    <Card
      title="Inspection Results"
      headerActions={
        <span className="text-sm text-gray-500">{formatDate(currentResult.timestamp)}</span>
      }
    >
      {summaryCard}
      <DefectList defects={currentResult.defects} />
    </Card>
  );
});

InspectionResults.displayName = 'InspectionResults';

export default InspectionResults;
