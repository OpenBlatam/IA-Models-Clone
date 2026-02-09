'use client';

import { getSeverityColor } from '@/lib/utils';
import type { Defect } from '@/lib/types';

interface DefectListProps {
  defects: Defect[];
}

const DefectList = ({ defects }: DefectListProps): JSX.Element => {
  if (defects.length === 0) {
    return (
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Defects</h3>
        <div className="text-center py-8 text-gray-500 bg-green-50 rounded-lg">
          <p className="font-medium">No defects detected</p>
          <p className="text-sm mt-1">Product quality is excellent</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Detected Defects ({defects.length})
      </h3>
      <div className="space-y-3">
        {defects.map((defect, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border ${getSeverityColor(
              defect.severity
            )}`}
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <h4 className="font-semibold text-gray-900">{defect.type}</h4>
                <p className="text-sm text-gray-600 mt-1">
                  {defect.description}
                </p>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-xs font-medium ${getSeverityColor(
                  defect.severity
                )}`}
              >
                {defect.severity.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-600 mt-2">
              <span>Confidence: {(defect.confidence * 100).toFixed(1)}%</span>
              <span>Area: {defect.area} px²</span>
              <span>
                Location: ({defect.location[0]}, {defect.location[1]})
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DefectList;

