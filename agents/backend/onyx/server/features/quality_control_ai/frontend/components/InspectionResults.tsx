'use client';

import { useState } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react';
import { useQualityControlStore } from '@/lib/store';
import {
  formatQualityScore,
  getQualityColor,
  getStatusLabel,
  getSeverityColor,
  formatDate,
} from '@/lib/utils';
import DefectList from './DefectList';

const InspectionResults = (): JSX.Element => {
  const { currentResult } = useQualityControlStore();

  if (!currentResult) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Inspection Results
        </h2>
        <div className="text-center py-12 text-gray-500">
          <Info className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p>No inspection results yet</p>
          <p className="text-sm mt-2">
            Start an inspection to see results here
          </p>
        </div>
      </div>
    );
  }

  const { summary, quality_score, defects, anomalies, objects } = currentResult;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">
          Inspection Results
        </h2>
        <span className="text-sm text-gray-500">
          {formatDate(currentResult.timestamp)}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">Quality Score</div>
          <div
            className={`text-3xl font-bold ${getQualityColor(quality_score)}`}
          >
            {formatQualityScore(quality_score)}
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">Status</div>
          <div className="text-2xl font-semibold text-gray-900">
            {getStatusLabel(summary.status)}
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">Defects</div>
          <div className="text-3xl font-bold text-gray-900">
            {defects.length}
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">Objects</div>
          <div className="text-3xl font-bold text-gray-900">
            {objects.length}
          </div>
        </div>
      </div>

      <div className="mb-6">
        <div
          className={`p-4 rounded-lg border-l-4 ${
            summary.status === 'excellent' || summary.status === 'good'
              ? 'bg-green-50 border-green-500'
              : summary.status === 'acceptable'
              ? 'bg-yellow-50 border-yellow-500'
              : 'bg-red-50 border-red-500'
          }`}
        >
          <div className="flex items-start space-x-3">
            {summary.status === 'excellent' || summary.status === 'good' ? (
              <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            ) : summary.status === 'acceptable' ? (
              <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
            ) : (
              <XCircle className="w-5 h-5 text-red-600 mt-0.5" />
            )}
            <div>
              <h3 className="font-semibold text-gray-900 mb-1">
                Recommendation
              </h3>
              <p className="text-gray-700">{summary.recommendation}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-sm text-blue-600 mb-1">Critical</div>
          <div className="text-2xl font-bold text-blue-900">
            {summary.severity_counts.critical}
          </div>
        </div>
        <div className="bg-orange-50 rounded-lg p-4">
          <div className="text-sm text-orange-600 mb-1">Severe</div>
          <div className="text-2xl font-bold text-orange-900">
            {summary.severity_counts.severe}
          </div>
        </div>
        <div className="bg-yellow-50 rounded-lg p-4">
          <div className="text-sm text-yellow-600 mb-1">Moderate</div>
          <div className="text-2xl font-bold text-yellow-900">
            {summary.severity_counts.moderate}
          </div>
        </div>
      </div>

      <DefectList defects={defects} />
    </div>
  );
};

export default InspectionResults;

