import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';

/**
 * Share utilities
 */

export const shareFile = async (uri: string, mimeType?: string): Promise<boolean> => {
  try {
    const isAvailable = await Sharing.isAvailableAsync();
    if (!isAvailable) {
      console.error('Sharing is not available on this device');
      return false;
    }

    await Sharing.shareAsync(uri, {
      mimeType,
      UTI: Platform.OS === 'ios' ? mimeType : undefined,
    });

    return true;
  } catch (error) {
    console.error('Error sharing file:', error);
    return false;
  }
};

export const shareImage = async (uri: string): Promise<boolean> => {
  return shareFile(uri, 'image/jpeg');
};

export const sharePDF = async (uri: string): Promise<boolean> => {
  return shareFile(uri, 'application/pdf');
};

export const shareText = async (text: string, title?: string): Promise<boolean> => {
  try {
    const { Share } = require('react-native');
    await Share.share({
      message: text,
      title,
    });
    return true;
  } catch (error) {
    console.error('Error sharing text:', error);
    return false;
  }
};

