'use client';

import { useQuery } from '@tanstack/react-query';
import { BarChart3, TrendingUp } from 'lucide-react';
import { qualityControlApi } from '@/lib/api';
import { useQualityControlStore } from '@/lib/store';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const StatisticsPanel = (): JSX.Element => {
  const { inspectionHistory } = useQualityControlStore();

  const { data: alertStats } = useQuery({
    queryKey: ['alert-statistics'],
    queryFn: () => qualityControlApi.getAlertStatistics(),
    refetchInterval: 5000,
  });

  const recentScores = inspectionHistory
    .slice(0, 10)
    .map((result) => ({
      score: result.quality_score,
      defects: result.defects_detected,
    }));

  const defectTypeCounts = inspectionHistory.reduce(
    (acc, result) => {
      result.defects.forEach((defect) => {
        acc[defect.type] = (acc[defect.type] || 0) + 1;
      });
      return acc;
    },
    {} as Record<string, number>
  );

  const chartData = Object.entries(defectTypeCounts).map(([type, count]) => ({
    type,
    count,
  }));

  const averageScore =
    inspectionHistory.length > 0
      ? inspectionHistory.reduce((sum, r) => sum + r.quality_score, 0) /
        inspectionHistory.length
      : 0;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Statistics</h2>

      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">Total Inspections</div>
            <div className="text-2xl font-bold text-gray-900">
              {inspectionHistory.length}
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">Average Score</div>
            <div className="text-2xl font-bold text-gray-900">
              {averageScore.toFixed(1)}
            </div>
          </div>
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
    </div>
  );
};

export default StatisticsPanel;

