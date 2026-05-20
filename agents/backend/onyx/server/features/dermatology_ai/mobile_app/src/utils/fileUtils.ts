import * as FileSystem from 'expo-file-system';

/**
 * File utilities
 */

/**
 * Check if file exists
 */
export const fileExists = async (uri: string): Promise<boolean> => {
  try {
    const info = await FileSystem.getInfoAsync(uri);
    return info.exists;
  } catch {
    return false;
  }
};

/**
 * Get file size
 */
export const getFileSize = async (uri: string): Promise<number> => {
  try {
    const info = await FileSystem.getInfoAsync(uri);
    return info.exists && 'size' in info ? info.size : 0;
  } catch {
    return 0;
  }
};

/**
 * Delete file
 */
export const deleteFile = async (uri: string): Promise<boolean> => {
  try {
    await FileSystem.deleteAsync(uri, { idempotent: true });
    return true;
  } catch {
    return false;
  }
};

/**
 * Read file as string
 */
export const readFile = async (uri: string): Promise<string | null> => {
  try {
    return await FileSystem.readAsStringAsync(uri);
  } catch {
    return null;
  }
};

/**
 * Write file
 */
export const writeFile = async (
  uri: string,
  content: string
): Promise<boolean> => {
  try {
    await FileSystem.writeAsStringAsync(uri, content);
    return true;
  } catch {
    return false;
  }
};

/**
 * Copy file
 */
export const copyFile = async (
  fromUri: string,
  toUri: string
): Promise<boolean> => {
  try {
    await FileSystem.copyAsync({ from: fromUri, to: toUri });
    return true;
  } catch {
    return false;
  }
};

/**
 * Move file
 */
export const moveFile = async (
  fromUri: string,
  toUri: string
): Promise<boolean> => {
  try {
    await FileSystem.moveAsync({ from: fromUri, to: toUri });
    return true;
  } catch {
    return false;
  }
};

/**
 * Get directory info
 */
export const getDirectoryInfo = async (
  uri: string
): Promise<FileSystem.FileInfo[] | null> => {
  try {
    return await FileSystem.readDirectoryAsync(uri);
  } catch {
    return null;
  }
};

/**
 * Create directory
 */
export const createDirectory = async (uri: string): Promise<boolean> => {
  try {
    await FileSystem.makeDirectoryAsync(uri, { intermediates: true });
    return true;
  } catch {
    return false;
  }
};

/**
 * Delete directory
 */
export const deleteDirectory = async (uri: string): Promise<boolean> => {
  try {
    await FileSystem.deleteAsync(uri, { idempotent: true });
    return true;
  } catch {
    return false;
  }
};

