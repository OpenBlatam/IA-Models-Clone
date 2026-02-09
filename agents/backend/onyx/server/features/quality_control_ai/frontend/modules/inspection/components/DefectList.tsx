'use client';

import { memo } from 'react';
import { CheckCircle } from 'lucide-react';
import type { Defect } from '../types';
import DefectItem from './DefectItem';
import EmptyState from '@/components/ui/EmptyState';

interface DefectListProps {
  defects: Defect[];
}

const DefectList = memo(({ defects }: DefectListProps): JSX.Element => {
  if (defects.length === 0) {
    return (
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Defects</h3>
        <EmptyState
          icon={CheckCircle}
          title="No defects detected"
          description="Product quality is excellent"
          className="bg-green-50 rounded-lg"
        />
      </div>
    );
  }

  return (
    <div className="mt-6" role="region" aria-label="Detected defects">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Detected Defects ({defects.length})
      </h3>
      <div className="space-y-3" role="list">
        {defects.map((defect, index) => (
          <DefectItem
            key={`${defect.type}-${defect.location[0]}-${defect.location[1]}-${index}`}
            defect={defect}
          />
        ))}
      </div>
    </div>
  );
});

DefectList.displayName = 'DefectList';

export default DefectList;
