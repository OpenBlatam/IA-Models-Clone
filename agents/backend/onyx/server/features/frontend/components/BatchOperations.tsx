'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCheckSquare, FiSquare, FiTrash2, FiDownload, FiX } from 'react-icons/fi';
import { showToast } from '@/lib/toast';

interface BatchOperationsProps {
  items: any[];
  onDelete: (ids: string[]) => void;
  onExport?: (ids: string[]) => void;
  getId: (item: any) => string;
  getLabel?: (item: any) => string;
}

export default function BatchOperations({
  items,
  onDelete,
  onExport,
  getId,
  getLabel,
}: BatchOperationsProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isSelectMode, setIsSelectMode] = useState(false);

  const toggleSelect = (id: string) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const selectAll = () => {
    if (selectedIds.size === items.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(items.map(getId)));
    }
  };

  const handleDelete = () => {
    if (selectedIds.size === 0) return;
    if (confirm(`¿Eliminar ${selectedIds.size} elemento(s)?`)) {
      onDelete(Array.from(selectedIds));
      setSelectedIds(new Set());
      setIsSelectMode(false);
      showToast(`${selectedIds.size} elemento(s) eliminado(s)`, 'success');
    }
  };

  const handleExport = () => {
    if (selectedIds.size === 0 || !onExport) return;
    onExport(Array.from(selectedIds));
    showToast(`${selectedIds.size} elemento(s) exportado(s)`, 'success');
  };

  if (!isSelectMode && selectedIds.size === 0) {
    return (
      <button
        onClick={() => setIsSelectMode(true)}
        className="btn btn-secondary"
      >
        <FiCheckSquare size={18} className="mr-2" />
        Seleccionar Múltiples
      </button>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 100, opacity: 0 }}
        className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 z-50"
      >
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <button
              onClick={selectAll}
              className="btn-icon"
              title="Seleccionar todos"
            >
              {selectedIds.size === items.length ? (
                <FiCheckSquare size={20} />
              ) : (
                <FiSquare size={20} />
              )}
            </button>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {selectedIds.size} seleccionado(s)
            </span>
          </div>

          <div className="flex items-center gap-2">
            {onExport && (
              <button
                onClick={handleExport}
                disabled={selectedIds.size === 0}
                className="btn btn-secondary text-sm"
              >
                <FiDownload size={16} className="mr-1" />
                Exportar
              </button>
            )}
            <button
              onClick={handleDelete}
              disabled={selectedIds.size === 0}
              className="btn btn-danger text-sm"
            >
              <FiTrash2 size={16} className="mr-1" />
              Eliminar
            </button>
            <button
              onClick={() => {
                setIsSelectMode(false);
                setSelectedIds(new Set());
              }}
              className="btn btn-secondary text-sm"
            >
              <FiX size={16} className="mr-1" />
              Cancelar
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


