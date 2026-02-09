'use client';

import { useState } from 'react';
import { Download, FileText, FileSpreadsheet, FileCode } from 'lucide-react';
import { qualityControlApi } from '@/lib/api';
import { useQualityControlStore } from '@/lib/store';
import { downloadBlob } from '@/lib/utils';
import type { ReportRequest } from '@/lib/types';

const ReportGenerator = (): JSX.Element => {
  const { currentResult } = useQualityControlStore();
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerateReport = async (
    format: 'json' | 'csv' | 'html'
  ): Promise<void> => {
    if (!currentResult) {
      alert('No inspection result available');
      return;
    }

    setIsGenerating(true);
    try {
      const request: ReportRequest = {
        inspection_result: currentResult,
        format,
        include_images: format === 'html',
        include_charts: format === 'html',
      };

      const blob = await qualityControlApi.generateReport(request);
      const extension = format === 'json' ? 'json' : format === 'csv' ? 'csv' : 'html';
      const filename = `quality-report-${Date.now()}.${extension}`;
      downloadBlob(blob, filename);
    } catch (error) {
      console.error('Failed to generate report:', error);
      alert('Failed to generate report');
    } finally {
      setIsGenerating(false);
    }
  };

  if (!currentResult) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Generate Report
      </h2>
      <div className="space-y-3">
        <button
          onClick={() => handleGenerateReport('json')}
          disabled={isGenerating}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
          tabIndex={0}
          aria-label="Generate JSON report"
        >
          <FileText className="w-4 h-4" />
          <span>JSON Report</span>
        </button>
        <button
          onClick={() => handleGenerateReport('csv')}
          disabled={isGenerating}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
          tabIndex={0}
          aria-label="Generate CSV report"
        >
          <FileSpreadsheet className="w-4 h-4" />
          <span>CSV Report</span>
        </button>
        <button
          onClick={() => handleGenerateReport('html')}
          disabled={isGenerating}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
          tabIndex={0}
          aria-label="Generate HTML report"
        >
          <FileCode className="w-4 h-4" />
          <span>HTML Report</span>
        </button>
      </div>
    </div>
  );
};

export default ReportGenerator;

