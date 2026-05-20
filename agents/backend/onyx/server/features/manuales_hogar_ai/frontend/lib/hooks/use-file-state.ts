import { useState, useCallback } from 'react';

interface UseFileStateReturn {
  files: File[];
  setFiles: (files: File[]) => void;
  addFiles: (newFiles: File[]) => void;
  removeFile: (index: number) => void;
  clearFiles: () => void;
  hasFiles: boolean;
  fileCount: number;
}

export const useFileState = (initialFiles: File[] = []): UseFileStateReturn => {
  const [files, setFiles] = useState<File[]>(initialFiles);

  const addFiles = useCallback((newFiles: File[]): void => {
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const removeFile = useCallback((index: number): void => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const clearFiles = useCallback((): void => {
    setFiles([]);
  }, []);

  return {
    files,
    setFiles,
    addFiles,
    removeFile,
    clearFiles,
    hasFiles: files.length > 0,
    fileCount: files.length,
  };
};

