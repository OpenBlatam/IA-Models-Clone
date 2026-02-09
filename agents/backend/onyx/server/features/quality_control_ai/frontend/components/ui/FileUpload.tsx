'use client';

import { memo, useCallback, useState } from 'react';
import { Upload, X, File } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { cn, formatFileSize, isValidImageFile, isValidVideoFile } from '@/lib/utils';
import { Button } from './Button';

interface FileUploadProps {
  onUpload: (files: File[]) => void;
  accept?: string | string[];
  multiple?: boolean;
  maxSize?: number;
  className?: string;
  label?: string;
  disabled?: boolean;
  showPreview?: boolean;
}

const FileUpload = memo(
  ({
    onUpload,
    accept,
    multiple = false,
    maxSize,
    className,
    label = 'Upload files',
    disabled = false,
    showPreview = false,
  }: FileUploadProps): JSX.Element => {
    const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

    const onDrop = useCallback(
      (acceptedFiles: File[]) => {
        setUploadedFiles((prev) => (multiple ? [...prev, ...acceptedFiles] : acceptedFiles));
        onUpload(acceptedFiles);
      },
      [onUpload, multiple]
    );

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      onDrop,
      accept: accept ? (Array.isArray(accept) ? accept : [accept]) : undefined,
      multiple,
      maxSize,
      disabled,
    });

    const removeFile = useCallback((index: number) => {
      setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
    }, []);

    return (
      <div className={cn('space-y-4', className)}>
        <div
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
            isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-gray-400',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" aria-hidden="true" />
          <p className="text-sm text-gray-600 mb-2">
            {isDragActive ? 'Drop files here' : `Drag & drop files here, or click to select`}
          </p>
          <p className="text-xs text-gray-500">
            {accept && `Accepted: ${Array.isArray(accept) ? accept.join(', ') : accept}`}
            {maxSize && ` • Max size: ${formatFileSize(maxSize)}`}
          </p>
        </div>

        {showPreview && uploadedFiles.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Uploaded files:</h4>
            {uploadedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3 flex-1">
                  {isValidImageFile(file) || isValidVideoFile(file) ? (
                    <div className="w-10 h-10 bg-primary-100 rounded flex items-center justify-center">
                      <File className="w-5 h-5 text-primary-600" aria-hidden="true" />
                    </div>
                  ) : (
                    <File className="w-5 h-5 text-gray-400" aria-hidden="true" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeFile(index)}
                  className="ml-2"
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="w-4 h-4" aria-hidden="true" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }
);

FileUpload.displayName = 'FileUpload';

export default FileUpload;

