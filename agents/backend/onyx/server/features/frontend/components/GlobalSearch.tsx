'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiSearch, FiX, FiFileText, FiCheckCircle, FiClock } from 'react-icons/fi';
import { useHotkeys } from 'react-hotkeys-hook';
import { apiClient } from '@/lib/api-client';
import { useAppStore } from '@/store/app-store';
import type { TaskListItem, DocumentListItem } from '@/types/api';

export default function GlobalSearch() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<{
    tasks: TaskListItem[];
    documents: DocumentListItem[];
  }>({ tasks: [], documents: [] });
  const [isSearching, setIsSearching] = useState(false);
  const { setActiveView } = useAppStore();

  useHotkeys('ctrl+k,cmd+k', () => {
    setIsOpen(true);
  }, { preventDefault: true });

  useEffect(() => {
    if (!isOpen) {
      setQuery('');
      setResults({ tasks: [], documents: [] });
    }
  }, [isOpen]);

  useEffect(() => {
    if (query.length < 2) {
      setResults({ tasks: [], documents: [] });
      return;
    }

    const search = async () => {
      setIsSearching(true);
      try {
        const [tasksResponse, documentsResponse] = await Promise.all([
          apiClient.listTasks({ search: query, limit: 5 }),
          apiClient.listDocuments(5, 0),
        ]);

        const filteredTasks = tasksResponse.tasks.filter((task) =>
          task.query_preview.toLowerCase().includes(query.toLowerCase()) ||
          task.task_id.toLowerCase().includes(query.toLowerCase())
        );

        const filteredDocuments = documentsResponse.documents.filter((doc) =>
          doc.query_preview.toLowerCase().includes(query.toLowerCase()) ||
          doc.task_id.toLowerCase().includes(query.toLowerCase())
        );

        setResults({
          tasks: filteredTasks,
          documents: filteredDocuments,
        });
      } catch (error) {
        console.error('Search error:', error);
      } finally {
        setIsSearching(false);
      }
    };

    const timeout = setTimeout(search, 300);
    return () => clearTimeout(timeout);
  }, [query]);

  const handleSelectTask = (taskId: string) => {
    setActiveView('tasks');
    setIsOpen(false);
    // Scroll to task could be implemented
  };

  const handleSelectDocument = (taskId: string) => {
    setActiveView('documents');
    setIsOpen(false);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-50"
            onClick={() => setIsOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed top-20 left-1/2 transform -translate-x-1/2 w-full max-w-2xl z-50 p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl overflow-hidden">
              <div className="flex items-center gap-3 p-4 border-b border-gray-200 dark:border-gray-700">
                <FiSearch className="text-gray-400" size={20} />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Buscar tareas, documentos..."
                  className="flex-1 bg-transparent border-none outline-none text-gray-900 dark:text-white placeholder-gray-400"
                  autoFocus
                />
                <button
                  onClick={() => setIsOpen(false)}
                  className="btn-icon"
                >
                  <FiX size={18} />
                </button>
              </div>

              <div className="max-h-96 overflow-y-auto">
                {isSearching ? (
                  <div className="p-8 text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
                  </div>
                ) : query.length < 2 ? (
                  <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                    <p>Escribe al menos 2 caracteres para buscar</p>
                  </div>
                ) : results.tasks.length === 0 && results.documents.length === 0 ? (
                  <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                    <p>No se encontraron resultados</p>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {results.tasks.length > 0 && (
                      <div>
                        <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase">
                          Tareas ({results.tasks.length})
                        </div>
                        {results.tasks.map((task) => (
                          <button
                            key={task.task_id}
                            onClick={() => handleSelectTask(task.task_id)}
                            className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-3"
                          >
                            <FiCheckCircle
                              className={`${
                                task.status === 'completed'
                                  ? 'text-green-500'
                                  : task.status === 'processing'
                                  ? 'text-yellow-500'
                                  : 'text-gray-400'
                              }`}
                              size={18}
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                                {task.query_preview}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                                {task.task_id}
                              </p>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}

                    {results.documents.length > 0 && (
                      <div>
                        <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase">
                          Documentos ({results.documents.length})
                        </div>
                        {results.documents.map((doc) => (
                          <button
                            key={doc.task_id}
                            onClick={() => handleSelectDocument(doc.task_id)}
                            className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-3"
                          >
                            <FiFileText className="text-primary-500" size={18} />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                                {doc.query_preview}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                                {doc.task_id}
                              </p>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-4">
                    <span>↑↓ Navegar</span>
                    <span>Enter Seleccionar</span>
                    <span>Esc Cerrar</span>
                  </div>
                  <kbd className="px-2 py-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded">
                    Ctrl+K
                  </kbd>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}


