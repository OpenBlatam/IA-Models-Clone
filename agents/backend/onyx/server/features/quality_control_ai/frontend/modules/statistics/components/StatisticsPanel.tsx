'use client';

import { memo, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useAlerts } from '@/modules/alerts/hooks/useAlerts';
import { useQualityControlStore } from '@/lib/store';
import { StatisticsService } from '@/lib/services';
import Card from '@/components/ui/Card';
import StatCard from '@/components/ui/StatCard';

const StatisticsPanel = memo((): JSX.Element => {
  const { inspectionHistory } = useQualityControlStore();
  const { statistics: alertStats } = useAlerts();

  const recentScores = useMemo(
    () => StatisticsService.getRecentScores(inspectionHistory, 10),
    [inspectionHistory]
  );

  const defectTypeCounts = useMemo(
    () => StatisticsService.getDefectTypeDistribution(inspectionHistory),
    [inspectionHistory]
  );

  const chartData = useMemo(
    () => Object.entries(defectTypeCounts).map(([type, count]) => ({ type, count })),
    [defectTypeCounts]
  );

  const averageScore = useMemo(
    () => StatisticsService.calculateAverageScore(inspectionHistory),
    [inspectionHistory]
  );

  return (
    <Card title="Statistics">
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <StatCard label="Total Inspections" value={inspectionHistory.length} />
          <StatCard label="Average Score" value={averageScore.toFixed(1)} />
        </div>

        {alertStats && (
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Alert Statistics
            </h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-600">Total: </span>
                <span className="font-semibold">{alertStats.total}</span>
              </div>
              <div>
                <span className="text-gray-600">Critical: </span>
                <span className="font-semibold text-red-600">
                  {alertStats.by_level.critical}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Warnings: </span>
                <span className="font-semibold text-yellow-600">
                  {alertStats.by_level.warning}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Recent Critical: </span>
                <span className="font-semibold text-red-600">
                  {alertStats.recent_critical}
                </span>
              </div>
            </div>
          </div>
        )}

        {chartData.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Defect Types
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="type"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={10}
                />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#0ea5e9" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {recentScores.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Recent Scores
            </h3>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={recentScores}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="defects" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="score" fill="#28a745" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </Card>
  );
});

StatisticsPanel.displayName = 'StatisticsPanel';

export default StatisticsPanel;
