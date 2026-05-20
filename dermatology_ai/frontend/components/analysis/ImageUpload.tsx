'use client';

import React, { useCallback, useState, memo } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';

interface ImageUploadProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number;
  currentFile?: File | null;
}

export const ImageUpload: React.FC<ImageUploadProps> = memo(({
  onFileSelect,
  accept = 'image/*',
  maxSize = 10 * 1024 * 1024, // 10MB
  currentFile,
}) => {
  const [preview, setPreview] = useState<string | null>(null);
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
        setPreview(URL.createObjectURL(file));
        onFileSelect(file);
      }
    },
    [onFileSelect, maxSize]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept ? { [accept]: [] } : undefined,
    maxFiles: 1,
    maxSize,
  });

  const removeFile = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
    setError(null);
    onFileSelect(null as any);
  };

  return (
    <div className="w-full">
      {!preview ? (
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
            transition-all duration-300 ease-out
            group overflow-hidden
            ${
              isDragActive
                ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 scale-[1.02] shadow-lg'
                : 'border-gray-300 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-600 hover:bg-gradient-to-br hover:from-gray-50 hover:to-blue-50/50 dark:hover:from-gray-900/50 dark:hover:to-blue-950/20 hover:shadow-xl'
            }
            ${error ? 'border-red-500 bg-red-50 dark:bg-red-950/20' : ''}
          `}
        >
          {/* Animated background gradient */}
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 bg-gradient-to-br from-blue-100/50 via-purple-100/50 to-pink-100/50 dark:from-blue-900/20 dark:via-purple-900/20 dark:to-pink-900/20" />
          
          <input {...getInputProps()} />
          <div className="relative z-10">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
              <Upload className="h-10 w-10 text-white" />
            </div>
            {isDragActive ? (
              <div className="space-y-2">
                <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                  Drop it here
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-lg font-semibold text-gray-700 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                  Drop image here or click
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  JPG, PNG, WEBP up to {maxSize / 1024 / 1024}MB
                </p>
              </div>
            )}
          </div>
        </div>
      ) : (
        <Card className="relative overflow-hidden border-0 shadow-xl bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm animate-fade-in">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl" />
            <img
              src={preview}
              alt="Preview"
              className="w-full h-auto rounded-xl max-h-96 object-contain relative z-10"
            />
            <button
              onClick={removeFile}
              className="absolute top-3 right-3 z-20 bg-red-500 text-white rounded-full p-2.5 hover:bg-red-600 hover:scale-110 transition-all duration-200 shadow-lg"
              aria-label="Remove image"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="mt-4 flex items-center justify-between p-2">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              <div className="p-1.5 rounded-lg bg-blue-100 dark:bg-blue-900/50">
                <ImageIcon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              </div>
              <span className="truncate max-w-xs">{currentFile?.name || 'Image selected'}</span>
            </div>
            <Button variant="outline" size="sm" onClick={removeFile} className="hover:bg-gray-100 dark:hover:bg-gray-800">
              Change
            </Button>
          </div>
        </Card>
      )}
      {error && (
        <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
          {error}
        </div>
      )}
    </div>
  );
});

ImageUpload.displayName = 'ImageUpload';

