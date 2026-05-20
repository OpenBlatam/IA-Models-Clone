export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

export function getFileExtension(filename: string): string {
  return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
}

export function getFileNameWithoutExtension(filename: string): string {
  return filename.replace(/\.[^/.]+$/, '');
}

export function isValidFileType(file: File, acceptedTypes: string[]): boolean {
  if (acceptedTypes.length === 0) return true;

  const fileExtension = getFileExtension(file.name).toLowerCase();
  const mimeType = file.type.toLowerCase();

  return acceptedTypes.some((type) => {
    const normalizedType = type.toLowerCase().trim();
    
    // Check MIME type
    if (normalizedType.includes('/')) {
      if (normalizedType.endsWith('/*')) {
        const baseType = normalizedType.split('/')[0];
        return mimeType.startsWith(baseType + '/');
      }
      return mimeType === normalizedType;
    }
    
    // Check file extension
    return fileExtension === normalizedType.replace('.', '');
  });
}

export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

export function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as ArrayBuffer);
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

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

