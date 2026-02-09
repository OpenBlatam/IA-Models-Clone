'use client';

import { memo, useMemo } from 'react';
import StatCard from '@/components/ui/StatCard';
import { AlertCircle, AlertTriangle, Info } from 'lucide-react';

interface SeverityCountsProps {
  counts: {
    critical: number;
    severe: number;
    moderate: number;
    minor?: number;
  };
}

const SeverityCounts = memo(({ counts }: SeverityCountsProps): JSX.Element => {
  const total = useMemo(
    () => counts.critical + counts.severe + counts.moderate + (counts.minor || 0),
    [counts]
  );

  if (total === 0) {
    return null;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6" role="group" aria-label="Defect severity counts">
      <StatCard
        label="Critical"
        value={counts.critical}
        className="bg-red-50 border-l-4 border-red-500"
        valueClassName="text-2xl font-bold text-red-900"
        icon={<AlertCircle className="w-4 h-4" aria-hidden="true" />}
        aria-label={`Critical defects: ${counts.critical}`}
      />
      <StatCard
        label="Severe"
        value={counts.severe}
        className="bg-orange-50 border-l-4 border-orange-500"
        valueClassName="text-2xl font-bold text-orange-900"
        icon={<AlertTriangle className="w-4 h-4" aria-hidden="true" />}
        aria-label={`Severe defects: ${counts.severe}`}
      />
      <StatCard
        label="Moderate"
        value={counts.moderate}
        className="bg-yellow-50 border-l-4 border-yellow-500"
        valueClassName="text-2xl font-bold text-yellow-900"
        icon={<Info className="w-4 h-4" aria-hidden="true" />}
        aria-label={`Moderate defects: ${counts.moderate}`}
      />
    </div>
  );
});

SeverityCounts.displayName = 'SeverityCounts';

export default SeverityCounts;

