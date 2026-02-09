'use client';

import { memo } from 'react';
import { getSeverityColor, formatPercentage } from '@/lib/utils';
import type { Defect } from '../types';
import Badge from '@/components/ui/Badge';
import { cn } from '@/lib/utils';

interface DefectItemProps {
  defect: Defect;
}

const DefectItem = memo(({ defect }: DefectItemProps): JSX.Element => {
  const severityColor = getSeverityColor(defect.severity);
  const badgeVariant = 
    defect.severity === 'critical' || defect.severity === 'severe' 
      ? 'error' 
      : defect.severity === 'moderate' 
      ? 'warning' 
      : 'info';

  return (
    <div
      className={cn('p-4 rounded-lg border transition-shadow hover:shadow-md', severityColor)}
      role="article"
      aria-label={`Defect: ${defect.type}, ${defect.severity} severity`}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900">{defect.type}</h4>
          {defect.description && (
            <p className="text-sm text-gray-600 mt-1">{defect.description}</p>
          )}
        </div>
        <Badge variant={badgeVariant} className="ml-2 flex-shrink-0">
          {defect.severity.toUpperCase()}
        </Badge>
      </div>
      <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 mt-3">
        <span aria-label={`Confidence: ${formatPercentage(defect.confidence)}`}>
          Confidence: <strong>{formatPercentage(defect.confidence)}</strong>
        </span>
        <span aria-label={`Area: ${defect.area} pixels squared`}>
          Area: <strong>{defect.area} px²</strong>
        </span>
        <span aria-label={`Location: X ${defect.location[0]}, Y ${defect.location[1]}`}>
          Location: <strong>({defect.location[0]}, {defect.location[1]})</strong>
        </span>
      </div>
    </div>
  );
});

DefectItem.displayName = 'DefectItem';

export default DefectItem;

