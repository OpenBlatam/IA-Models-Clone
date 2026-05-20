'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiClock, FiGitBranch, FiEye, FiRotateCcw, FiX } from 'react-icons/fi';
import { format } from 'date-fns';
import { apiClient } from '@/lib/api-client';
import { showToast } from '@/lib/toast';

interface DocumentVersion {
  id: string;
  taskId: string;
  version: number;
  content: string;
  createdAt: Date;
  metadata?: {
    query?: string;
    business_area?: string;
    document_type?: string;
  };
}

interface DocumentHistoryProps {
  taskId: string;
  isOpen: boolean;
  onClose: () => void;
  onRestore?: (version: DocumentVersion) => void;
}

export default function DocumentHistory({
  taskId,
  isOpen,
  onClose,
  onRestore,
}: DocumentHistoryProps) {
  const [versions, setVersions] = useState<DocumentVersion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState<DocumentVersion | null>(null);

  useEffect(() => {
    if (isOpen && taskId) {
      loadHistory();
    }
  }, [isOpen, taskId]);

  const loadHistory = async () => {
    setIsLoading(true);
    try {
      const stored = localStorage.getItem(`bul_document_history_${taskId}`);
      if (stored) {
        const history = JSON.parse(stored).map((v: any) => ({
          ...v,
          createdAt: new Date(v.createdAt),
        }));
        setVersions(history.sort((a: DocumentVersion, b: DocumentVersion) => 
          b.version - a.version
        ));
      } else {
        // Load current version as v1
        try {
          const doc = await apiClient.getTaskDocument(taskId);
          if (doc?.document?.content) {
            const currentVersion: DocumentVersion = {
              id: `${taskId}_v1`,
              taskId,
              version: 1,
              content: doc.document.content,
              createdAt: new Date(),
              metadata: doc.metadata,
            };
            setVersions([currentVersion]);
            saveVersion(currentVersion);
          }
        } catch (error) {
          console.error('Error loading document:', error);
        }
      }
    } catch (error) {
      console.error('Error loading history:', error);
      showToast('Error al cargar historial', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const saveVersion = (version: DocumentVersion) => {
    const stored = localStorage.getItem(`bul_document_history_${taskId}`);
    const history = stored ? JSON.parse(stored) : [];
    history.push(version);
    // Keep only last 20 versions
    const limited = history.slice(-20);
    localStorage.setItem(`bul_document_history_${taskId}`, JSON.stringify(limited));
  };

  const createNewVersion = async () => {
    try {
      const doc = await apiClient.getTaskDocument(taskId);
      if (doc?.document?.content) {
        const newVersion: DocumentVersion = {
          id: `${taskId}_v${versions.length + 1}`,
          taskId,
          version: versions.length + 1,
          content: doc.document.content,
          createdAt: new Date(),
          metadata: doc.metadata,
        };
        const updated = [newVersion, ...versions];
        setVersions(updated);
        saveVersion(newVersion);
        showToast('Nueva versión guardada', 'success');
      }
    } catch (error) {
      showToast('Error al crear versión', 'error');
    }
  };

  const handleRestore = (version: DocumentVersion) => {
    if (onRestore) {
      onRestore(version);
    }
    showToast(`Versión ${version.version} restaurada`, 'success');
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FiGitBranch size={24} className="text-primary-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Historial de Versiones
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={createNewVersion}
                className="btn btn-secondary text-sm"
                title="Crear nueva versión"
              >
                <FiRotateCcw size={16} className="mr-1" />
                Nueva Versión
              </button>
              <button onClick={onClose} className="btn-icon">
                <FiX size={20} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {isLoading ? (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
                <p className="text-gray-600 dark:text-gray-400">Cargando historial...</p>
              </div>
            ) : versions.length === 0 ? (
              <div className="text-center py-12">
                <FiGitBranch size={48} className="mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600 dark:text-gray-400">No hay versiones guardadas</p>
              </div>
            ) : (
              <div className="space-y-3">
                {versions.map((version, index) => (
                  <motion.div
                    key={version.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedVersion?.id === version.id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700'
                    }`}
                    onClick={() => setSelectedVersion(version)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="px-2 py-1 bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 text-xs font-medium rounded">
                            v{version.version}
                          </span>
                          {index === 0 && (
                            <span className="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 text-xs font-medium rounded">
                              Actual
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                          {version.metadata?.query?.substring(0, 100) || 'Sin descripción'}...
                        </p>
                        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-500">
                          <span className="flex items-center gap-1">
                            <FiClock size={12} />
                            {format(version.createdAt, 'PPp')}
                          </span>
                          <span>
                            {version.content.length.toLocaleString()} caracteres
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 ml-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedVersion(version);
                          }}
                          className="btn-icon text-primary-600"
                          title="Ver versión"
                        >
                          <FiEye size={18} />
                        </button>
                        {index > 0 && onRestore && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRestore(version);
                            }}
                            className="btn-icon text-green-600"
                            title="Restaurar versión"
                          >
                            <FiRotateCcw size={18} />
                          </button>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {selectedVersion && (
            <div className="p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 max-h-64 overflow-y-auto">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Vista Previa - Versión {selectedVersion.version}
              </h4>
              <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono">
                {selectedVersion.content.substring(0, 500)}
                {selectedVersion.content.length > 500 && '...'}
              </pre>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


