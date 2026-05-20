import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

// Pure functions for file system operations

export async function readFile(uri: string): Promise<string> {
  try {
    return await FileSystem.readAsStringAsync(uri);
  } catch (error) {
    console.error('Error reading file:', error);
    throw error;
  }
}

export async function writeFile(
  uri: string,
  contents: string
): Promise<void> {
  try {
    await FileSystem.writeAsStringAsync(uri, contents);
  } catch (error) {
    console.error('Error writing file:', error);
    throw error;
  }
}

export async function deleteFile(uri: string): Promise<void> {
  try {
    const fileInfo = await FileSystem.getInfoAsync(uri);
    if (fileInfo.exists) {
      await FileSystem.deleteAsync(uri);
    }
  } catch (error) {
    console.error('Error deleting file:', error);
    throw error;
  }
}

export async function shareFile(uri: string, mimeType?: string): Promise<void> {
  try {
    const isAvailable = await Sharing.isAvailableAsync();
    if (!isAvailable) {
      throw new Error('Sharing is not available on this device');
    }

    await Sharing.shareAsync(uri, {
      mimeType,
      dialogTitle: 'Compartir archivo',
    });
  } catch (error) {
    console.error('Error sharing file:', error);
    throw error;
  }
}

export async function getFileInfo(uri: string) {
  try {
    return await FileSystem.getInfoAsync(uri);
  } catch (error) {
    console.error('Error getting file info:', error);
    throw error;
  }
}

export function getDocumentDirectory(): string {
  return FileSystem.documentDirectory || '';
}

export function getCacheDirectory(): string {
  return FileSystem.cacheDirectory || '';
}

