import { useState, useCallback } from 'react';

interface UseImageUploadOptions {
  maxSize?: number; // in MB
  onSuccess?: (url: string) => void;
  onError?: (error: string) => void;
}

export const useImageUpload = ({ maxSize = 5, onSuccess, onError }: UseImageUploadOptions = {}) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback(
    async (file: File) => {
      if (file.size > maxSize * 1024 * 1024) {
        const errorMsg = `El archivo es demasiado grande. Tamaño máximo: ${maxSize}MB`;
        setError(errorMsg);
        if (onError) {
          onError(errorMsg);
        }
        return;
      }

      if (!file.type.startsWith('image/')) {
        const errorMsg = 'El archivo debe ser una imagen';
        setError(errorMsg);
        if (onError) {
          onError(errorMsg);
        }
        return;
      }

      setIsUploading(true);
      setError(null);

      try {
        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result as string;
          setPreview(result);
          setIsUploading(false);
          if (onSuccess) {
            onSuccess(result);
          }
        };
        reader.onerror = () => {
          const errorMsg = 'Error al leer el archivo';
          setError(errorMsg);
          setIsUploading(false);
          if (onError) {
            onError(errorMsg);
          }
        };
        reader.readAsDataURL(file);
      } catch (err) {
        const errorMsg = 'Error al procesar la imagen';
        setError(errorMsg);
        setIsUploading(false);
        if (onError) {
          onError(errorMsg);
        }
      }
    },
    [maxSize, onSuccess, onError]
  );

  const reset = useCallback(() => {
    setPreview(null);
    setError(null);
    setIsUploading(false);
  }, []);

  return {
    preview,
    isUploading,
    error,
    handleFileSelect,
    reset,
  };
};

