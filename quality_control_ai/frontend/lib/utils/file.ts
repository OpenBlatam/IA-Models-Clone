export const getFileExtension = (filename: string): string => {
  return filename.split('.').pop()?.toLowerCase() || '';
};

export const getFileName = (filepath: string): string => {
  return filepath.split('/').pop() || filepath;
};

export const getFileSize = (file: File): number => {
  return file.size;
};

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
};

export const isValidImageFile = (file: File): boolean => {
  return file.type.startsWith('image/');
};

export const isValidVideoFile = (file: File): boolean => {
  return file.type.startsWith('video/');
};

export const isValidAudioFile = (file: File): boolean => {
  return file.type.startsWith('audio/');
};

export const isValidPdfFile = (file: File): boolean => {
  return file.type === 'application/pdf';
};

export const isValidTextFile = (file: File): boolean => {
  return file.type.startsWith('text/');
};

export const readFileAsText = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
};

export const readFileAsDataURL = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export const readFileAsArrayBuffer = (file: File): Promise<ArrayBuffer> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
};

export const downloadFile = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};

export const createFileFromText = (text: string, filename: string, mimeType = 'text/plain'): File => {
  const blob = new Blob([text], { type: mimeType });
  return new File([blob], filename, { type: mimeType });
};

