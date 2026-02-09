'use client';

import { useRef, useState, useCallback, memo } from 'react';
import { Upload, X } from 'lucide-react';
import { useInspection } from '../hooks/useInspection';
import { useAsyncOperation } from '@/lib/hooks/useAsyncOperation';
import { Button } from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import { cn } from '@/lib/utils';
import { readFileAsBase64, readFileAsDataURL, isValidImageFile } from '@/lib/utils/dom';

const ImageUpload = memo((): JSX.Element => {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { inspectFrame } = useInspection();

  const { execute: handleFileSelect, isLoading } = useAsyncOperation(
    async (file: File) => {
      if (!isValidImageFile(file)) {
        throw new Error('Please select an image file');
      }

      const previewUrl = await readFileAsDataURL(file);
      setPreview(previewUrl);

      const base64 = await readFileAsBase64(file);
      const result = await inspectFrame(base64);
      if (!result) {
        throw new Error('Failed to inspect image');
      }
      return result;
    },
    {
      successMessage: 'Image inspected successfully',
      errorMessage: 'Failed to inspect image',
    }
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>): void => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFileSelect(file);
      }
    },
    [handleFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((): void => {
    setIsDragging(false);
  }, []);

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>): void => {
      const file = e.target.files?.[0];
      if (file) {
        handleFileSelect(file);
      }
    },
    [handleFileSelect]
  );

  const handleClick = useCallback((): void => {
    fileInputRef.current?.click();
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent): void => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleClick();
      }
    },
    [handleClick]
  );

  const handleClear = useCallback((): void => {
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  return (
    <Card title="Upload Image">
      <div
        className={cn(
          'relative border-2 border-dashed rounded-lg p-8 text-center transition-colors',
          isDragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400',
          isLoading && 'opacity-50 pointer-events-none'
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        role="button"
        tabIndex={0}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        aria-label="Upload image for inspection"
        aria-disabled={isLoading}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInputChange}
          className="hidden"
          aria-label="File input"
          disabled={isLoading}
        />
        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="max-h-64 mx-auto rounded-lg mb-4"
              loading="lazy"
            />
            <Button
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              variant="danger"
              size="sm"
              className="absolute top-2 right-2"
              aria-label="Remove image"
              tabIndex={0}
            >
              <X className="w-4 h-4" aria-hidden="true" />
            </Button>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" aria-hidden="true" />
            <p className="text-gray-600 mb-2">
              Drag and drop an image here, or click to select
            </p>
            <p className="text-sm text-gray-500">Supports JPG, PNG, GIF</p>
          </>
        )}
      </div>
    </Card>
  );
});

ImageUpload.displayName = 'ImageUpload';

export default ImageUpload;
