'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiX, FiFileText, FiGitCompare } from 'react-icons/fi';
import { apiClient } from '@/lib/api-client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { showToast } from '@/lib/toast';

interface DocumentComparisonProps {
  taskIds: string[];
  onClose: () => void;
}

export default function DocumentComparison({ taskIds, onClose }: DocumentComparisonProps) {
  const [documents, setDocuments] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadDocuments = async () => {
      try {
        const docs = await Promise.all(
          taskIds.map((id) => apiClient.getTaskDocument(id))
        );
        setDocuments(docs);
      } catch (error: any) {
        showToast(error.message || 'Error al cargar documentos', 'error');
      } finally {
        setIsLoading(false);
      }
    };

    loadDocuments();
  }, [taskIds]);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-7xl w-full max-h-[90vh] flex flex-col"
      >
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <FiGitCompare size={24} className="text-primary-600" />
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Comparar Documentos
            </h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <FiX size={24} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {documents.map((doc, index) => (
              <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-4">
                  <FiFileText size={20} className="text-primary-600" />
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    Documento {index + 1}
                  </h4>
                  <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                    {taskIds[index]}
                  </span>
                </div>
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {doc?.document?.content || 'No hay contenido'}
                  </ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 border-t border-gray-200 dark:border-gray-700">
          <button onClick={onClose} className="btn btn-primary w-full">
            Cerrar
          </button>
        </div>
      </motion.div>
    </div>
  );
}


