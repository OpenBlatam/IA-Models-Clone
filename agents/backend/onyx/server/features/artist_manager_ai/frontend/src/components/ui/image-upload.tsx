'use client';

import { useState, useCallback } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dropzone } from '@/components/ui/dropzone';
import { cn } from '@/lib/utils';

interface ImageUploadProps {
  value?: string;
  onChange?: (url: string) => void;
  onRemove?: () => void;
  className?: string;
  maxSize?: number; // in MB
  accept?: string;
}

const ImageUpload = ({
  value,
  onChange,
  onRemove,
  className,
  maxSize = 5,
  accept = 'image/*',
}: ImageUploadProps) => {
  const [preview, setPreview] = useState<string | null>(value || null);
  const [isUploading, setIsUploading] = useState(false);

  const handleDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      if (file.size > maxSize * 1024 * 1024) {
        alert(`El archivo es demasiado grande. Tamaño máximo: ${maxSize}MB`);
        return;
      }

      setIsUploading(true);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setPreview(result);
        if (onChange) {
          onChange(result);
        }
        setIsUploading(false);
      };
      reader.readAsDataURL(file);
    },
    [maxSize, onChange]
  );

  const handleRemove = () => {
    setPreview(null);
    if (onRemove) {
      onRemove();
    }
  };

  if (preview) {
    return (
      <div className={cn('relative inline-block', className)}>
        <img
          src={preview}
          alt="Preview"
          className="w-full h-48 object-cover rounded-lg border border-gray-300"
        />
        <button
          onClick={handleRemove}
          className="absolute top-2 right-2 p-1 bg-red-600 text-white rounded-full hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
          aria-label="Eliminar imagen"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <Dropzone
      onDrop={handleDrop}
      accept={accept}
      maxSize={maxSize * 1024 * 1024}
      className={className}
      disabled={isUploading}
    >
      <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 transition-colors">
        {isUploading ? (
          <div className="text-sm text-gray-600">Subiendo...</div>
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-400 mb-4" />
            <p className="text-sm text-gray-600 mb-2">
              Arrastra una imagen aquí o haz clic para seleccionar
            </p>
            <p className="text-xs text-gray-500">Tamaño máximo: {maxSize}MB</p>
          </>
        )}
      </div>
    </Dropzone>
  );
};

export { ImageUpload };

