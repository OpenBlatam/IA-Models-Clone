import { useState } from 'react';
import * as FileSystem from 'expo-file-system';
import * as DocumentPicker from 'expo-document-picker';
import * as Sharing from 'expo-sharing';
import { Alert, Platform } from 'react-native';

export function useFilePicker() {
  const [isPicking, setIsPicking] = useState(false);

  const pickFile = async (options?: {
    type?: string[];
    copyToCacheDirectory?: boolean;
  }) => {
    try {
      setIsPicking(true);
      const result = await DocumentPicker.getDocumentAsync({
        type: options?.type || ['text/*', 'application/*'],
        copyToCacheDirectory: options?.copyToCacheDirectory ?? true,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        return result.assets[0];
      }
      return null;
    } catch (error) {
      console.error('Error picking file:', error);
      Alert.alert('Error', 'Failed to pick file');
      return null;
    } finally {
      setIsPicking(false);
    }
  };

  return { pickFile, isPicking };
}

export function useFileDownload() {
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState(0);

  const downloadFile = async (
    uri: string,
    filename: string,
    onProgress?: (progress: number) => void
  ) => {
    try {
      setIsDownloading(true);
      setProgress(0);

      const fileUri = `${FileSystem.documentDirectory}${filename}`;

      const downloadResumable = FileSystem.createDownloadResumable(
        uri,
        fileUri,
        {},
        (downloadProgress) => {
          const progressPercent =
            downloadProgress.totalBytesWritten /
            downloadProgress.totalBytesExpectedToWrite;
          setProgress(progressPercent);
          onProgress?.(progressPercent);
        }
      );

      const result = await downloadResumable.downloadAsync();

      if (result) {
        return result.uri;
      }
      return null;
    } catch (error) {
      console.error('Error downloading file:', error);
      Alert.alert('Error', 'Failed to download file');
      return null;
    } finally {
      setIsDownloading(false);
      setProgress(0);
    }
  };

  return { downloadFile, isDownloading, progress };
}

export function useFileShare() {
  const [isSharing, setIsSharing] = useState(false);

  const shareFile = async (uri: string, options?: { mimeType?: string; dialogTitle?: string }) => {
    try {
      setIsSharing(true);

      const isAvailable = await Sharing.isAvailableAsync();
      if (!isAvailable) {
        Alert.alert('Error', 'Sharing is not available on this device');
        return false;
      }

      await Sharing.shareAsync(uri, {
        mimeType: options?.mimeType,
        dialogTitle: options?.dialogTitle,
        UTI: Platform.OS === 'ios' ? options?.mimeType : undefined,
      });

      return true;
    } catch (error) {
      console.error('Error sharing file:', error);
      Alert.alert('Error', 'Failed to share file');
      return false;
    } finally {
      setIsSharing(false);
    }
  };

  return { shareFile, isSharing };
}

export function useFileDelete() {
  const deleteFile = async (uri: string) => {
    try {
      const fileInfo = await FileSystem.getInfoAsync(uri);
      if (fileInfo.exists) {
        await FileSystem.deleteAsync(uri, { idempotent: true });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error deleting file:', error);
      return false;
    }
  };

  return { deleteFile };
}

export function useFileInfo() {
  const getFileInfo = async (uri: string) => {
    try {
      const fileInfo = await FileSystem.getInfoAsync(uri);
      return fileInfo;
    } catch (error) {
      console.error('Error getting file info:', error);
      return null;
    }
  };

  return { getFileInfo };
}


