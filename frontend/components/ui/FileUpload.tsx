'use client';

import { ReactNode } from 'react';
import { FiUpload, FiFile, FiX } from 'react-icons/fi';
import { useDragAndDrop } from '@/hooks';
import { cn } from '@/utils/classNames';
import { Button } from './Button';

interface FileUploadProps {
  onUpload: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // in bytes
  disabled?: boolean;
  label?: string;
  description?: string;
  className?: string;
  children?: ReactNode;
}

export function FileUpload({
  onUpload,
  accept,
  multiple = false,
  maxSize,
  disabled = false,
  label = 'Arrastra archivos aquí o haz clic para seleccionar',
  description,
  className,
  children,
}: FileUploadProps) {
  const {
    isDragOver,
    dragProps,
    fileInputRef,
    openFileDialog,
    handleFileSelect,
  } = useDragAndDrop({
    onDrop: (files) => {
      if (maxSize) {
        const validFiles = files.filter((file) => file.size <= maxSize);
        if (validFiles.length !== files.length) {
          // Some files were too large
          console.warn('Some files exceed the maximum size');
        }
        if (validFiles.length > 0) {
          onUpload(validFiles);
        }
      } else {
        onUpload(files);
      }
    },
    accept,
    multiple,
    disabled,
  });

  return (
    <div className={cn('w-full', className)}>
      <div
        {...dragProps}
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
          isDragOver
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
            : 'border-gray-300 dark:border-gray-600',
          disabled && 'opacity-50 cursor-not-allowed',
          !disabled && 'cursor-pointer hover:border-primary-400'
        )}
        onClick={!disabled ? openFileDialog : undefined}
      >
        {children || (
          <>
            <FiUpload
              size={48}
              className={cn(
                'mx-auto mb-4',
                isDragOver ? 'text-primary-500' : 'text-gray-400'
              )}
            />
            <p className="text-sm font-medium text-gray-900 dark:text-white mb-1">
              {label}
            </p>
            {description && (
              <p className="text-xs text-gray-500 dark:text-gray-400">{description}</p>
            )}
            {maxSize && (
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
                Tamaño máximo: {(maxSize / 1024 / 1024).toFixed(2)} MB
              </p>
            )}
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled}
      />
    </div>
  );
}

interface FileListProps {
  files: File[];
  onRemove?: (index: number) => void;
  className?: string;
}

export function FileList({ files, onRemove, className }: FileListProps) {
  if (files.length === 0) return null;

  return (
    <div className={cn('mt-4 space-y-2', className)}>
      {files.map((file, index) => (
        <div
          key={`${file.name}-${index}`}
          className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
        >
          <FiFile size={20} className="text-gray-400 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
              {file.name}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {(file.size / 1024).toFixed(2)} KB
            </p>
          </div>
          {onRemove && (
            <button
              onClick={() => onRemove(index)}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
              aria-label="Eliminar archivo"
            >
              <FiX size={18} className="text-gray-400" />
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

