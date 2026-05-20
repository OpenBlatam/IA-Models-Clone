'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiDownload, FiX, FiFileText, FiImage, FiCode } from 'react-icons/fi';
import { exportToHTML } from '@/lib/export-utils';
import { exportToPDF } from '@/lib/pdf-export';

interface ExportAdvancedProps {
  content: string;
  filename: string;
  isOpen: boolean;
  onClose: () => void;
}

export default function ExportAdvanced({
  content,
  filename,
  isOpen,
  onClose,
}: ExportAdvancedProps) {
  const [format, setFormat] = useState<'md' | 'txt' | 'html' | 'pdf' | 'json' | 'xml'>('md');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeTimestamp, setIncludeTimestamp] = useState(true);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      let exportContent = content;
      let mimeType = 'text/plain';
      let extension = 'txt';

      if (includeTimestamp) {
        exportContent = `<!-- Generated: ${new Date().toISOString()} -->\n\n${exportContent}`;
      }

      if (includeMetadata) {
        const metadata = `<!-- Metadata: ${JSON.stringify({ format, exported: new Date().toISOString() })} -->\n\n`;
        exportContent = metadata + exportContent;
      }

      switch (format) {
        case 'md':
          mimeType = 'text/markdown';
          extension = 'md';
          break;
        case 'html':
          exportToHTML(content, filename);
          setIsExporting(false);
          return;
        case 'pdf':
          await exportToPDF(content, filename);
          setIsExporting(false);
          return;
        case 'json':
          exportContent = JSON.stringify({ content, metadata: { exported: new Date().toISOString() } }, null, 2);
          mimeType = 'application/json';
          extension = 'json';
          break;
        case 'xml':
          exportContent = `<?xml version="1.0" encoding="UTF-8"?>\n<document>\n<content><![CDATA[${content}]]></content>\n</document>`;
          mimeType = 'application/xml';
          extension = 'xml';
          break;
      }

      const blob = new Blob([exportContent], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.${extension}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error: any) {
      console.error('Export error:', error);
    } finally {
      setIsExporting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-md w-full"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FiDownload size={24} className="text-primary-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Exportación Avanzada
              </h3>
            </div>
            <button onClick={onClose} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Formato
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'md', label: 'Markdown', icon: FiFileText },
                  { value: 'txt', label: 'Texto', icon: FiFileText },
                  { value: 'html', label: 'HTML', icon: FiCode },
                  { value: 'pdf', label: 'PDF', icon: FiFileText },
                  { value: 'json', label: 'JSON', icon: FiCode },
                  { value: 'xml', label: 'XML', icon: FiCode },
                ].map((fmt) => (
                  <button
                    key={fmt.value}
                    onClick={() => setFormat(fmt.value as any)}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      format === fmt.value
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <fmt.icon size={20} className="mx-auto mb-1 text-gray-600 dark:text-gray-400" />
                    <span className="text-xs text-gray-700 dark:text-gray-300">{fmt.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <label className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Incluir Metadatos
                </span>
                <input
                  type="checkbox"
                  checked={includeMetadata}
                  onChange={(e) => setIncludeMetadata(e.target.checked)}
                  className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                />
              </label>
              <label className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Incluir Timestamp
                </span>
                <input
                  type="checkbox"
                  checked={includeTimestamp}
                  onChange={(e) => setIncludeTimestamp(e.target.checked)}
                  className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                />
              </label>
            </div>
          </div>

          <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex gap-3">
            <button onClick={onClose} className="btn btn-secondary flex-1">
              Cancelar
            </button>
            <button
              onClick={handleExport}
              disabled={isExporting}
              className="btn btn-primary flex-1"
            >
              {isExporting ? 'Exportando...' : 'Exportar'}
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


