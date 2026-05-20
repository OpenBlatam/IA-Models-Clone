/**
 * Image manipulation utilities
 */

import { Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export interface ImageInfo {
  uri: string;
  width: number;
  height: number;
  type?: string;
  name?: string;
  size?: number;
}

export async function getImageInfo(uri: string): Promise<ImageInfo | null> {
  return new Promise((resolve) => {
    Image.getSize(
      uri,
      (width, height) => {
        resolve({
          uri,
          width,
          height,
        });
      },
      (error) => {
        console.error('Error getting image size:', error);
        resolve(null);
      }
    );
  });
}

export async function pickImage(
  options?: ImagePicker.ImagePickerOptions
): Promise<ImagePicker.ImagePickerResult> {
  return await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    allowsEditing: true,
    aspect: [16, 9],
    quality: 0.8,
    ...options,
  });
}

export async function pickMultipleImages(
  options?: ImagePicker.ImagePickerOptions
): Promise<ImagePicker.ImagePickerResult> {
  return await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    allowsMultipleSelection: true,
    quality: 0.8,
    ...options,
  });
}

export async function takePhoto(
  options?: ImagePicker.ImagePickerOptions
): Promise<ImagePicker.ImagePickerResult> {
  return await ImagePicker.launchCameraAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    allowsEditing: true,
    aspect: [16, 9],
    quality: 0.8,
    ...options,
  });
}

export function calculateAspectRatio(width: number, height: number): number {
  return width / height;
}

export function calculateDimensions(
  originalWidth: number,
  originalHeight: number,
  maxWidth: number,
  maxHeight: number
): { width: number; height: number } {
  const aspectRatio = calculateAspectRatio(originalWidth, originalHeight);
  
  let width = originalWidth;
  let height = originalHeight;
  
  if (width > maxWidth) {
    width = maxWidth;
    height = width / aspectRatio;
  }
  
  if (height > maxHeight) {
    height = maxHeight;
    width = height * aspectRatio;
  }
  
  return { width, height };
}

export function getImageType(uri: string): string {
  const extension = uri.split('.').pop()?.toLowerCase();
  const imageTypes: Record<string, string> = {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    gif: 'image/gif',
    webp: 'image/webp',
    bmp: 'image/bmp',
  };
  
  return imageTypes[extension || ''] || 'image/jpeg';
}

export function isValidImageFormat(uri: string): boolean {
  const validFormats = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'];
  const extension = uri.split('.').pop()?.toLowerCase();
  return extension ? validFormats.includes(extension) : false;
}

export function formatImageSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function getImageDimensions(uri: string): Promise<{ width: number; height: number } | null> {
  return new Promise((resolve) => {
    Image.getSize(
      uri,
      (width, height) => resolve({ width, height }),
      () => resolve(null)
    );
  });
}

export function isImageLandscape(width: number, height: number): boolean {
  return width > height;
}

export function isImagePortrait(width: number, height: number): boolean {
  return height > width;
}

export function isImageSquare(width: number, height: number): boolean {
  return width === height;
}


