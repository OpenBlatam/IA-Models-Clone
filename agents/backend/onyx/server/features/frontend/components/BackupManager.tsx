'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiDownload, FiUpload, FiTrash2, FiClock, FiCheck } from 'react-icons/fi';
import { format } from 'date-fns';

interface Backup {
  id: string;
  name: string;
  timestamp: Date;
  size: number;
  data: any;
}

export default function BackupManager() {
  const [backups, setBackups] = useState<Backup[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    loadBackups();
  }, []);

  const loadBackups = () => {
    const stored = localStorage.getItem('bul_backups');
    if (stored) {
      setBackups(
        JSON.parse(stored).map((b: any) => ({
          ...b,
          timestamp: new Date(b.timestamp),
        }))
      );
    }
  };

  const createBackup = () => {
    const data = {
      favorites: localStorage.getItem('bul_favorites'),
      notes: localStorage.getItem('bul_quick_notes'),
      settings: localStorage.getItem('bul_settings'),
      notifications: localStorage.getItem('bul_notifications'),
      searchHistory: localStorage.getItem('bul_search_history'),
      autosave: localStorage.getItem('bul_generate_autosave'),
    };

    const backup: Backup = {
      id: Date.now().toString(),
      name: `Backup ${format(new Date(), 'PPp')}`,
      timestamp: new Date(),
      size: JSON.stringify(data).length,
      data,
    };

    const updated = [backup, ...backups].slice(0, 10); // Keep last 10
    setBackups(updated);
    localStorage.setItem('bul_backups', JSON.stringify(updated));
  };

  const restoreBackup = (backup: Backup) => {
    if (confirm('¿Restaurar este backup? Esto sobrescribirá tus datos actuales.')) {
      Object.entries(backup.data).forEach(([key, value]) => {
        if (value) {
          localStorage.setItem(key, value as string);
        }
      });
      alert('Backup restaurado exitosamente');
      window.location.reload();
    }
  };

  const deleteBackup = (id: string) => {
    const updated = backups.filter((b) => b.id !== id);
    setBackups(updated);
    localStorage.setItem('bul_backups', JSON.stringify(updated));
  };

  const downloadBackup = (backup: Backup) => {
    const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bul-backup-${backup.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const uploadBackup = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const backup = JSON.parse(e.target?.result as string);
        const updated = [backup, ...backups].slice(0, 10);
        setBackups(updated);
        localStorage.setItem('bul_backups', JSON.stringify(updated));
        alert('Backup importado exitosamente');
      } catch (error) {
        alert('Error al importar backup');
      }
    };
    reader.readAsText(file);
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-32 right-4 bg-purple-600 text-white p-4 rounded-full shadow-lg hover:bg-purple-700 transition-colors z-40"
        title="Gestión de Backups"
      >
        <FiDownload size={24} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Gestión de Backups
                </h3>
                <button onClick={() => setIsOpen(false)} className="btn-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18" />
                    <path d="M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                <div className="flex gap-3">
                  <button onClick={createBackup} className="btn btn-primary flex-1">
                    <FiDownload size={18} className="mr-2" />
                    Crear Backup
                  </button>
                  <label className="btn btn-secondary flex-1 cursor-pointer">
                    <FiUpload size={18} className="mr-2" />
                    Importar
                    <input
                      type="file"
                      accept=".json"
                      onChange={uploadBackup}
                      className="hidden"
                    />
                  </label>
                </div>

                <div className="space-y-2">
                  {backups.length === 0 ? (
                    <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                      <FiClock size={48} className="mx-auto mb-2 opacity-50" />
                      <p>No hay backups guardados</p>
                    </div>
                  ) : (
                    backups.map((backup) => (
                      <motion.div
                        key={backup.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-center justify-between"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <FiCheck size={16} className="text-green-500" />
                            <span className="font-medium text-gray-900 dark:text-white">
                              {backup.name}
                            </span>
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {format(backup.timestamp, 'PPp')} • {(backup.size / 1024).toFixed(2)} KB
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => restoreBackup(backup)}
                            className="btn btn-secondary text-sm"
                            title="Restaurar"
                          >
                            Restaurar
                          </button>
                          <button
                            onClick={() => downloadBackup(backup)}
                            className="btn-icon"
                            title="Descargar"
                          >
                            <FiDownload size={18} />
                          </button>
                          <button
                            onClick={() => deleteBackup(backup.id)}
                            className="btn-icon text-red-600"
                            title="Eliminar"
                          >
                            <FiTrash2 size={18} />
                          </button>
                        </div>
                      </motion.div>
                    ))
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}


