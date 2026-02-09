'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiClock, FiFileText, FiGitBranch } from 'react-icons/fi';
import { format } from 'date-fns';
import { apiClient } from '@/lib/api-client';
import DocumentModal from './DocumentModal';

interface Version {
  taskId: string;
  createdAt: string;
  query: string;
  status: string;
}

interface DocumentVersionsProps {
  baseQuery: string;
}

export default function DocumentVersions({ baseQuery }: DocumentVersionsProps) {
  const [versions, setVersions] = useState<Version[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadVersions = async () => {
      try {
        const response = await apiClient.listTasks({ limit: 20 });
        const filtered = response.tasks
          .filter((task) =>
            task.query_preview.toLowerCase().includes(baseQuery.toLowerCase().substring(0, 20))
          )
          .map((task) => ({
            taskId: task.task_id,
            createdAt: task.created_at,
            query: task.query_preview,
            status: task.status,
          }))
          .slice(0, 5);

        setVersions(filtered);
      } catch (error) {
        console.error('Error loading versions:', error);
      } finally {
        setIsLoading(false);
      }
    };

    if (baseQuery) {
      loadVersions();
    }
  }, [baseQuery]);

  if (isLoading || versions.length === 0) return null;

  return (
    <div className="card mt-4">
      <div className="flex items-center gap-2 mb-4">
        <FiGitBranch size={20} className="text-primary-600" />
        <h3 className="font-semibold text-gray-900 dark:text-white">Versiones Relacionadas</h3>
      </div>
      <div className="space-y-2">
        {versions.map((version, index) => (
          <motion.button
            key={version.taskId}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            onClick={() => setSelectedVersion(version.taskId)}
            className="w-full text-left p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                  {version.query.substring(0, 60)}...
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <FiClock size={14} className="text-gray-400" />
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {format(new Date(version.createdAt), "PPp")}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      version.status === 'completed'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                    }`}
                  >
                    {version.status}
                  </span>
                </div>
              </div>
              <FiFileText size={18} className="text-primary-500 flex-shrink-0" />
            </div>
          </motion.button>
        ))}
      </div>

      {selectedVersion && (
        <DocumentModal taskId={selectedVersion} onClose={() => setSelectedVersion(null)} />
      )}
    </div>
  );
}


