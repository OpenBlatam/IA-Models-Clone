/**
 * File utilities
 */

import { formatFileSize } from './format';

// Read file as text
export async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

// Read file as data URL
export async function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Read file as array buffer
export async function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as ArrayBuffer);
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

// Download file
export function downloadFile(data: Blob | string, filename: string, mimeType?: string) {
  const blob = typeof data === 'string' ? new Blob([data], { type: mimeType || 'text/plain' }) : data;
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Get file extension
export function getFileExtension(filename: string): string {
  return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
}

// Get file name without extension
export function getFileNameWithoutExtension(filename: string): string {
  return filename.replace(/\.[^/.]+$/, '');
}

// Validate file type
export function validateFileType(file: File, allowedTypes: string[]): boolean {
  return allowedTypes.some((type) => {
    if (type.endsWith('/*')) {
      return file.type.startsWith(type.slice(0, -1));
    }
    return file.type === type;
  });
}

// Validate file size
export function validateFileSize(file: File, maxSize: number): boolean {
  return file.size <= maxSize;
}

// Format file info
export function formatFileInfo(file: File): string {
  return `${file.name} (${formatFileSize(file.size)})`;
}

// Create file from text
export function createFileFromText(text: string, filename: string, mimeType: string = 'text/plain'): File {
  const blob = new Blob([text], { type: mimeType });
  return new File([blob], filename, { type: mimeType });
}



