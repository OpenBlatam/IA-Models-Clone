import * as Clipboard from 'expo-clipboard';
import { Platform } from 'react-native';

export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await Clipboard.setStringAsync(text);
    return true;
  } catch (error) {
    console.error('Error copying to clipboard:', error);
    return false;
  }
};

export const getFromClipboard = async (): Promise<string | null> => {
  try {
    const text = await Clipboard.getStringAsync();
    return text || null;
  } catch (error) {
    console.error('Error getting from clipboard:', error);
    return null;
  }
};

export const hasClipboardContent = async (): Promise<boolean> => {
  try {
    const text = await Clipboard.getStringAsync();
    return text.length > 0;
  } catch (error) {
    return false;
  }
};

