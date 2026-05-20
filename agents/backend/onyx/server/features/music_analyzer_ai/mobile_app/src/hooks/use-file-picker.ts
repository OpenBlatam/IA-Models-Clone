import { useState, useCallback } from 'react';
import * as DocumentPicker from 'expo-document-picker';
import { Alert } from 'react-native';

interface FileResult {
  uri: string;
  name: string;
  size: number;
  mimeType: string | null;
}

/**
 * Hook for file picking functionality
 * Handles document selection
 */
export function useFilePicker() {
  const [file, setFile] = useState<FileResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const pickDocument = useCallback(
    async (options: {
      type?: string[];
      copyToCacheDirectory?: boolean;
      multiple?: boolean;
    } = {}) => {
      setIsLoading(true);
      try {
        const result = await DocumentPicker.getDocumentAsync({
          type: options.type,
          copyToCacheDirectory: options.copyToCacheDirectory ?? true,
          multiple: options.multiple ?? false,
        });

        if (!result.canceled && result.assets && result.assets.length > 0) {
          const asset = result.assets[0];
          const fileResult: FileResult = {
            uri: asset.uri,
            name: asset.name,
            size: asset.size || 0,
            mimeType: asset.mimeType || null,
          };
          setFile(fileResult);
          return fileResult;
        }
        return null;
      } catch (error) {
        console.error('File picker error:', error);
        Alert.alert('Error', 'Failed to pick file');
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clearFile = useCallback(() => {
    setFile(null);
  }, []);

  return {
    file,
    isLoading,
    pickDocument,
    clearFile,
  };
}

