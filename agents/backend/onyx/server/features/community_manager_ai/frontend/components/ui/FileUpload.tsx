'use client';

import { useCallback } from 'react';
import { useDropzone, FileRejection } from 'react-dropzone';
import { Upload, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './Button';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  className?: string;
  disabled?: boolean;
}

export const FileUpload = ({
  onFileSelect,
  accept = { 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'] },
  maxSize = 10 * 1024 * 1024, // 10MB
  className,
  disabled = false,
}: FileUploadProps) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
    disabled,
  });

  return (
    <div className={cn('w-full', className)}>
      <div
        {...getRootProps()}
        className={cn(
          'flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors',
          'cursor-pointer hover:border-primary-500 hover:bg-primary-50',
          isDragActive && 'border-primary-500 bg-primary-50',
          disabled && 'cursor-not-allowed opacity-50',
          fileRejections.length > 0 && 'border-red-500'
        )}
        aria-label="Subir archivo"
        tabIndex={disabled ? -1 : 0}
      >
        <input {...getInputProps()} />
        <Upload className={cn('h-8 w-8 mb-2', isDragActive ? 'text-primary-600' : 'text-gray-400')} />
        <p className="text-sm text-gray-600 text-center">
          {isDragActive ? 'Suelta el archivo aquí' : 'Arrastra un archivo o haz clic para seleccionar'}
        </p>
        <p className="text-xs text-gray-500 mt-1">PNG, JPG, GIF hasta {maxSize / 1024 / 1024}MB</p>
      </div>
      {fileRejections.length > 0 && (
        <div className="mt-2 text-sm text-red-600">
          {fileRejections.map((rejection: FileRejection) => {
            const { file, errors } = rejection;
            return (
            <div key={file.name}>
              {errors.map((error: { code: string; message: string }) => (
                <p key={error.code}>
                  {error.code === 'file-too-large'
                    ? 'El archivo es demasiado grande'
                    : error.code === 'file-invalid-type'
                    ? 'Tipo de archivo no válido'
                    : 'Error al subir el archivo'}
                </p>
              ))}
            </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

