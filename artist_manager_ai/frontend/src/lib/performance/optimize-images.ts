export const optimizeImageUrl = (url: string, width?: number, quality = 80): string => {
  if (!url) return url;
  
  // Si es una URL externa, puedes usar un servicio de optimización de imágenes
  // Por ejemplo, usando un CDN o servicio como Cloudinary, Imgix, etc.
  
  // Ejemplo básico: agregar parámetros de optimización si el servicio lo soporta
  const urlObj = new URL(url);
  
  if (width) {
    urlObj.searchParams.set('w', width.toString());
  }
  urlObj.searchParams.set('q', quality.toString());
  urlObj.searchParams.set('auto', 'format');
  
  return urlObj.toString();
};

export const getImagePlaceholder = (width = 400, height = 300): string => {
  return `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='${width}' height='${height}'%3E%3Crect fill='%23f3f4f6' width='${width}' height='${height}'/%3E%3C/svg%3E`;
};

export const preloadImage = (src: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = src;
  });
};

