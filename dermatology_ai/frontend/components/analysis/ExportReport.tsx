'use client';

import React, { useState } from 'react';
import { Button } from '../ui/Button';
import { Modal } from '../ui/Modal';
import { apiClient } from '@/lib/api/client';
import { Download, FileText, FileJson, File } from 'lucide-react';
import toast from 'react-hot-toast';

interface ExportReportProps {
  analysisId: string;
  analysisData?: any;
}

export const ExportReport: React.FC<ExportReportProps> = ({
  analysisId,
  analysisData,
}) => {
  const [isExporting, setIsExporting] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const handleExportJson = async () => {
    setIsExporting(true);
    try {
      const response = await apiClient.generateReportJson(analysisId);
      if (response.success && response.report_data) {
        const dataStr = JSON.stringify(response.report_data, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `analisis-${analysisId}.json`;
        link.click();
        URL.revokeObjectURL(url);
        toast.success('JSON report downloaded');
        setShowModal(false);
      }
    } catch (error: any) {
      toast.error(error.message || 'Error exporting JSON');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportPdf = async () => {
    setIsExporting(true);
    try {
      const blob = await apiClient.generateReportPdf(analysisId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `analisis-${analysisId}.pdf`;
      link.click();
      URL.revokeObjectURL(url);
      toast.success('PDF report downloaded');
      setShowModal(false);
    } catch (error: any) {
      toast.error(error.message || 'Error exporting PDF');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportHtml = async () => {
    setIsExporting(true);
    try {
      const response = await apiClient.generateReportHtml(analysisId);
      if (response.success && response.report_data) {
        const htmlContent = typeof response.report_data === 'string'
          ? response.report_data
          : JSON.stringify(response.report_data, null, 2);
        const dataBlob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `analisis-${analysisId}.html`;
        link.click();
        URL.revokeObjectURL(url);
        toast.success('HTML report downloaded');
        setShowModal(false);
      }
    } catch (error: any) {
      toast.error(error.message || 'Error exporting HTML');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportJsonDirect = () => {
    if (!analysisData) {
      toast.error('No data to export');
      return;
    }
    const dataStr = JSON.stringify(analysisData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analisis-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Data exported as JSON');
  };

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setShowModal(true)}
        className="flex items-center space-x-2"
      >
        <Download className="h-4 w-4" />
        <span>Export Report</span>
      </Button>

      <Modal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        title="Export Report"
        size="md"
      >
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            Choose your preferred export format
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={handleExportPdf}
              disabled={isExporting}
              className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FileText className="h-8 w-8 text-primary-600 mx-auto mb-2" />
              <p className="font-medium text-gray-900">PDF</p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Formatted document
              </p>
            </button>

            <button
              onClick={handleExportJson}
              disabled={isExporting}
              className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FileJson className="h-8 w-8 text-primary-600 mx-auto mb-2" />
              <p className="font-medium text-gray-900">JSON</p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Structured data
              </p>
            </button>

            <button
              onClick={handleExportHtml}
              disabled={isExporting}
              className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <File className="h-8 w-8 text-primary-600 mx-auto mb-2" />
              <p className="font-medium text-gray-900">HTML</p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Web page
              </p>
            </button>
          </div>

          {analysisData && (
            <div className="pt-4 border-t">
              <button
                onClick={handleExportJsonDirect}
                className="w-full p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm text-gray-700"
              >
                Export current data as JSON
              </button>
            </div>
          )}

          {isExporting && (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600 mx-auto mb-2"></div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Generating report...</p>
            </div>
          )}
        </div>
      </Modal>
    </>
  );
};

