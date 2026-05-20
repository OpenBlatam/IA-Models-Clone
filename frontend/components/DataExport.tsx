'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiDownload, FiDatabase, FiX } from 'react-icons/fi';

interface DataExportProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function DataExport({ isOpen, onClose }: DataExportProps) {
  const [exportType, setExportType] = useState<'all' | 'favorites' | 'notes' | 'settings'>('all');
  const [isExporting, setIsExporting] = useState(false);

  const exportData = () => {
    setIsExporting(true);
    
    let data: any = {};

    switch (exportType) {
      case 'all':
        data = {
          favorites: localStorage.getItem('bul_favorites'),
          notes: localStorage.getItem('bul_quick_notes'),
          settings: localStorage.getItem('bul_settings'),
          notifications: localStorage.getItem('bul_notifications'),
          searchHistory: localStorage.getItem('bul_search_history'),
          autosave: localStorage.getItem('bul_generate_autosave'),
          comments: Object.keys(localStorage)
            .filter((key) => key.startsWith('bul_comments_'))
            .reduce((acc, key) => {
              acc[key] = localStorage.getItem(key);
              return acc;
            }, {} as Record<string, string | null>),
        };
        break;
      case 'favorites':
        data = { favorites: localStorage.getItem('bul_favorites') };
        break;
      case 'notes':
        data = { notes: localStorage.getItem('bul_quick_notes') };
        break;
      case 'settings':
        data = { settings: localStorage.getItem('bul_settings') };
        break;
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bul-export-${exportType}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setIsExporting(false);
    onClose();
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
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-md w-full"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FiDatabase size={24} className="text-primary-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Exportar Datos
              </h3>
            </div>
            <button onClick={onClose} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Tipo de Exportación
              </label>
              <select
                value={exportType}
                onChange={(e) => setExportType(e.target.value as any)}
                className="select w-full"
              >
                <option value="all">Todos los Datos</option>
                <option value="favorites">Solo Favoritos</option>
                <option value="notes">Solo Notas</option>
                <option value="settings">Solo Configuración</option>
              </select>
            </div>

            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <p className="text-xs text-blue-800 dark:text-blue-200">
                Los datos se exportarán en formato JSON. Puedes importarlos más tarde desde la
                gestión de backups.
              </p>
            </div>
          </div>

          <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex gap-3">
            <button onClick={onClose} className="btn btn-secondary flex-1">
              Cancelar
            </button>
            <button
              onClick={exportData}
              disabled={isExporting}
              className="btn btn-primary flex-1"
            >
              <FiDownload size={18} className="mr-2" />
              {isExporting ? 'Exportando...' : 'Exportar'}
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


