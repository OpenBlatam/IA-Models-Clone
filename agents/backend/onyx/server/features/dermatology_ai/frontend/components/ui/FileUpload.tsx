'use client';

import React, { useCallback, useState, memo } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X } from 'lucide-react';
import { clsx } from 'clsx';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  multiple?: boolean;
  currentFile?: File | null;
  label?: string;
  description?: string;
}

export const FileUpload: React.FC<FileUploadProps> = memo(({
  onFileSelect,
  accept,
  maxSize = 10 * 1024 * 1024,
  multiple = false,
  currentFile,
  label = 'Upload file',
  description,
}) => {
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        if (file.size > maxSize) {
          setError(`File is too large. Maximum ${maxSize / 1024 / 1024}MB`);
          return;
        }
        setError(null);
        onFileSelect(file);
      }
    },
    [onFileSelect, maxSize]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: multiple ? undefined : 1,
    maxSize,
  });

  const removeFile = useCallback(() => {
    setError(null);
    onFileSelect(null as any);
  }, [onFileSelect]);

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          {label}
        </label>
      )}
      {description && (
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
          {description}
        </p>
      )}

      {!currentFile ? (
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
            isDragActive
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
              : 'border-gray-300 dark:border-gray-700 hover:border-primary-400 hover:bg-gray-50 dark:hover:bg-gray-800',
            error && 'border-red-500 bg-red-50 dark:bg-red-900/20'
          )}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500 mb-4" />
          {isDragActive ? (
            <p className="text-primary-600 dark:text-primary-400 font-medium">
              Drop file here
            </p>
          ) : (
            <>
              <p className="text-gray-600 dark:text-gray-400 mb-2">
                Drag file here or click to browse
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Maximum size: {maxSize / 1024 / 1024}MB
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <File className="h-5 w-5 text-primary-600 dark:text-primary-400" />
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {currentFile.name}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {(currentFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <button
            onClick={removeFile}
            className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
            aria-label="Remove file"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {error && (
        <div className="mt-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 p-2 rounded">
          {error}
        </div>
      )}
    </div>
  );
});

FileUpload.displayName = 'FileUpload';


