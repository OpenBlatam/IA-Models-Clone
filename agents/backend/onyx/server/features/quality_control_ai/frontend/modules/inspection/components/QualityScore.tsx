'use client';

import { memo } from 'react';
import { formatQualityScore, getQualityColor } from '@/lib/utils';
import StatCard from '@/components/ui/StatCard';
import { cn } from '@/lib/utils';

interface QualityScoreProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
}

const QualityScore = memo(({ score, size = 'md' }: QualityScoreProps): JSX.Element => {
  const sizeClasses = {
    sm: 'text-2xl',
    md: 'text-3xl',
    lg: 'text-4xl',
  };

  return (
    <StatCard
      label="Quality Score"
      value={formatQualityScore(score)}
      valueClassName={cn(sizeClasses[size], 'font-bold', getQualityColor(score))}
      aria-label={`Quality score: ${formatQualityScore(score)}`}
    />
  );
});

QualityScore.displayName = 'QualityScore';

export default QualityScore;

