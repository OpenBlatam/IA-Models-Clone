'use client';

import { memo, useCallback } from 'react';
import { FileText, FileSpreadsheet, FileCode } from 'lucide-react';
import { reportsApi } from '../api';
import { useQualityControlStore } from '@/lib/store';
import { downloadBlob } from '@/lib/utils/dom';
import { useAsyncOperation } from '@/lib/hooks/useAsyncOperation';
import type { ReportRequest } from '../types';
import { Button } from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import EmptyState from '@/components/ui/EmptyState';

const ReportGenerator = memo((): JSX.Element => {
  const { currentResult } = useQualityControlStore();

  const getReportExtension = useCallback(
    (format: 'json' | 'csv' | 'html'): string => {
      return format === 'json' ? 'json' : format === 'csv' ? 'csv' : 'html';
    },
    []
  );

  const { execute: handleGenerateReport, isLoading: isGenerating } = useAsyncOperation(
    async (format: 'json' | 'csv' | 'html') => {
      if (!currentResult) {
        throw new Error('No inspection result available');
      }

      const request: ReportRequest = {
        inspection_result: currentResult,
        format,
        include_images: format === 'html',
        include_charts: format === 'html',
      };

      const blob = await reportsApi.generate(request);
      const extension = getReportExtension(format);
      const filename = `quality-report-${Date.now()}.${extension}`;
      downloadBlob(blob, filename);
      return blob;
    },
    {
      successMessage: 'Report generated successfully',
      errorMessage: 'Failed to generate report',
    }
  );

  if (!currentResult) {
    return (
      <Card title="Generate Report">
        <EmptyState title="No inspection result available" />
      </Card>
    );
  }

  return (
    <Card title="Generate Report">
      <div className="space-y-3">
        <Button
          onClick={() => handleGenerateReport('json')}
          disabled={isGenerating}
          variant="secondary"
          className="w-full"
          tabIndex={0}
          aria-label="Generate JSON report"
        >
          <FileText className="w-4 h-4" aria-hidden="true" />
          <span>JSON Report</span>
        </Button>
        <Button
          onClick={() => handleGenerateReport('csv')}
          disabled={isGenerating}
          variant="secondary"
          className="w-full"
          tabIndex={0}
          aria-label="Generate CSV report"
        >
          <FileSpreadsheet className="w-4 h-4" aria-hidden="true" />
          <span>CSV Report</span>
        </Button>
        <Button
          onClick={() => handleGenerateReport('html')}
          disabled={isGenerating}
          variant="secondary"
          className="w-full"
          tabIndex={0}
          aria-label="Generate HTML report"
        >
          <FileCode className="w-4 h-4" aria-hidden="true" />
          <span>HTML Report</span>
        </Button>
      </div>
    </Card>
  );
});

ReportGenerator.displayName = 'ReportGenerator';

export default ReportGenerator;
