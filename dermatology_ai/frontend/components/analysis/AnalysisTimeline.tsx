'use client';

import React from 'react';
import { Timeline } from '../ui/Timeline';
import { format } from 'date-fns';
import { HistoryRecord } from '@/lib/types/api';

interface AnalysisTimelineProps {
  records: HistoryRecord[];
  onSelectRecord?: (record: HistoryRecord) => void;
}

export const AnalysisTimeline: React.FC<AnalysisTimelineProps> = ({
  records,
  onSelectRecord,
}) => {
  const timelineItems = records
    .slice()
    .reverse()
    .map((record, index) => ({
      id: record.record_id,
      title: `Analysis #${record.record_id.slice(0, 8)}`,
      description: `Score: ${record.analysis.quality_scores.overall_score.toFixed(1)} | Type: ${record.analysis.skin_type}`,
      date: format(new Date(record.timestamp), 'd MMM yyyy'),
      status: index === 0 ? ('current' as const) : ('completed' as const),
    }));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Analysis History
      </h3>
      <Timeline items={timelineItems} />
    </div>
  );
};


