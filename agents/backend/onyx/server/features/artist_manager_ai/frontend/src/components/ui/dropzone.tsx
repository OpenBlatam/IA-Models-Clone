'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface DropzoneProps {
  onFileAccepted: (file: File) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  label?: string;
  error?: string;
  className?: string;
}

const Dropzone = ({ onFileAccepted, accept, maxSize = 5242880, label, error, className }: DropzoneProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const selectedFile = acceptedFiles[0];
        setFile(selectedFile);
        onFileAccepted(selectedFile);

        if (selectedFile.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = () => {
            setPreview(reader.result as string);
          };
          reader.readAsDataURL(selectedFile);
        }
      }
    },
    [onFileAccepted]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
  });

  const handleRemove = () => {
    setFile(null);
    setPreview(null);
  };

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      )}
      {!file ? (
        <div
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
            isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400',
            error && 'border-red-500'
          )}
          tabIndex={0}
          role="button"
          aria-label="Subir archivo"
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-sm text-gray-600 mb-2">
            {isDragActive ? 'Suelta el archivo aquí' : 'Arrastra y suelta un archivo aquí'}
          </p>
          <p className="text-xs text-gray-500">o haz clic para seleccionar</p>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative border border-gray-300 rounded-lg p-4"
        >
          {preview ? (
            <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded-lg mb-2" />
          ) : (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-700">{file.name}</span>
            </div>
          )}
          <button
            onClick={handleRemove}
            className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500"
            aria-label="Eliminar archivo"
            tabIndex={0}
          >
            <X className="w-4 h-4" />
          </button>
        </motion.div>
      )}
      {error && (
        <p className="mt-1 text-sm text-red-600" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export { Dropzone };

