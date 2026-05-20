'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { FiUpload, FiFile, FiX } from 'react-icons/fi';
import { showToast } from '@/lib/toast';

interface DragDropZoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number; // in MB
  className?: string;
}

export default function DragDropZone({
  onFileSelect,
  accept = '.txt,.md,.doc,.docx',
  maxSize = 10,
  className = '',
}: DragDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        if (file.size > maxSize * 1024 * 1024) {
          showToast(`El archivo es demasiado grande. Máximo ${maxSize}MB`, 'error');
          return;
        }
        setSelectedFile(file);
        onFileSelect(file);
        showToast('Archivo cargado exitosamente', 'success');
      }
    },
    [maxSize, onFileSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        if (file.size > maxSize * 1024 * 1024) {
          showToast(`El archivo es demasiado grande. Máximo ${maxSize}MB`, 'error');
          return;
        }
        setSelectedFile(file);
        onFileSelect(file);
        showToast('Archivo cargado exitosamente', 'success');
      }
    },
    [maxSize, onFileSelect]
  );

  const readFileContent = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  return (
    <div className={className}>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
            : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800'
        }`}
      >
        {selectedFile ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center justify-between p-3 bg-white dark:bg-gray-700 rounded-lg"
          >
            <div className="flex items-center gap-3">
              <FiFile size={24} className="text-primary-600" />
              <div className="text-left">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
            </div>
            <button
              onClick={() => {
                setSelectedFile(null);
                showToast('Archivo eliminado', 'info');
              }}
              className="btn-icon text-red-600"
            >
              <FiX size={18} />
            </button>
          </motion.div>
        ) : (
          <>
            <FiUpload
              size={48}
              className={`mx-auto mb-4 ${
                isDragging ? 'text-primary-600' : 'text-gray-400'
              }`}
            />
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              Arrastra un archivo aquí o{' '}
              <label className="text-primary-600 dark:text-primary-400 cursor-pointer hover:underline">
                selecciona uno
                <input
                  type="file"
                  accept={accept}
                  onChange={handleFileInput}
                  className="hidden"
                />
              </label>
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Formatos: {accept} (Máximo {maxSize}MB)
            </p>
          </>
        )}
      </div>
    </div>
  );
}


