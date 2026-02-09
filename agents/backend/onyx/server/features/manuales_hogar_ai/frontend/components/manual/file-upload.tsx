'use client';

import { useState } from 'react';
import { Button } from '../ui/button';
import { Upload, X } from 'lucide-react';
import { FILE_UPLOAD, MESSAGES } from '@/lib/constants';
import { generateUniqueId } from '@/lib/utils/id-generator';
import toast from 'react-hot-toast';
import type { FileUploadProps } from '@/lib/types/components';

export const FileUpload = ({
  files,
  onFilesChange,
  maxFiles = FILE_UPLOAD.MAX_IMAGES,
  accept = 'image/*',
  label = 'Haz clic o arrastra imágenes aquí',
  multiple = true,
}: FileUploadProps): JSX.Element => {
  const [inputId] = useState(() => generateUniqueId('file-upload'));

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    if (!event.target.files) return;

    const newFiles = Array.from(event.target.files);
    
    if (newFiles.length + files.length > maxFiles) {
      toast.error(MESSAGES.FILE.MAX_FILES(maxFiles, multiple));
      return;
    }

    const maxSize = FILE_UPLOAD.MAX_SIZE_MB * 1024 * 1024;
    const invalidFiles = newFiles.filter(file => file.size > maxSize);
    
    if (invalidFiles.length > 0) {
      toast.error(MESSAGES.FILE.MAX_SIZE(FILE_UPLOAD.MAX_SIZE_MB));
      return;
    }

    if (multiple) {
      onFilesChange([...files, ...newFiles].slice(0, maxFiles));
    } else {
      onFilesChange(newFiles.slice(0, 1));
    }
  };

  const handleRemoveFile = (index: number): void => {
    onFilesChange(files.filter((_, i) => i !== index));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLLabelElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      document.getElementById(inputId)?.click();
    }
  };

  return (
    <div>
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
        <input
          type="file"
          id={inputId}
          accept={accept}
          multiple={multiple}
          onChange={handleFileChange}
          className="hidden"
        />
        <label
          htmlFor={inputId}
          className="cursor-pointer flex flex-col items-center justify-center"
          tabIndex={0}
          onKeyDown={handleKeyDown}
          aria-label={label}
        >
          <Upload className="h-12 w-12 text-gray-400 mb-4" />
          <span className="text-sm text-gray-600">{label}</span>
        </label>
      </div>
      {files.length > 0 && (
        <div className="mt-4 space-y-2">
          {files.map((file, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-2 bg-gray-100 rounded"
            >
              <span className="text-sm">{file.name}</span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => handleRemoveFile(index)}
                aria-label={`Eliminar ${file.name}`}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

