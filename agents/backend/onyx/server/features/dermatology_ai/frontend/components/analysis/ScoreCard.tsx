'use client';

import React, { memo, useMemo } from 'react';
import { Card } from '../ui/Card';
import { getScoreColor, getScoreBgColor } from '@/lib/utils/scoreUtils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface ScoreCardProps {
  label: string;
  score: number;
  showTrend?: boolean;
  className?: string;
}

export const ScoreCard: React.FC<ScoreCardProps> = memo(({
  label,
  score,
  showTrend = true,
  className,
}) => {
  const trendIcon = useMemo(() => {
    if (!showTrend) return null;
    if (score >= 80) return <TrendingUp className="h-4 w-4 text-green-600 dark:text-green-400" />;
    if (score >= 60) return <Minus className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />;
    return <TrendingDown className="h-4 w-4 text-red-600 dark:text-red-400" />;
  }, [showTrend, score]);

  return (
    <Card className={`p-4 ${className || ''}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
        {trendIcon}
      </div>
      <div className="flex items-center space-x-2">
        <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${getScoreBgColor(score)}`}
            style={{ width: `${score}%` }}
          />
        </div>
        <span className={`text-lg font-bold ${getScoreColor(score)}`}>
          {score.toFixed(1)}
        </span>
      </div>
    </Card>
  );
});

ScoreCard.displayName = 'ScoreCard';

