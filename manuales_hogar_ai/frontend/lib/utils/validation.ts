import { z } from 'zod';

export const manualDescriptionSchema = z
  .string()
  .min(10, 'La descripción debe tener al menos 10 caracteres')
  .max(1000, 'La descripción no puede exceder 1000 caracteres')
  .trim();

export const ratingSchema = z.object({
  rating: z.number().min(1, 'La calificación debe ser al menos 1').max(5, 'La calificación no puede ser mayor a 5'),
  comment: z.string().max(500, 'El comentario no puede exceder 500 caracteres').optional(),
});

export const searchQuerySchema = z
  .string()
  .min(1, 'La búsqueda debe tener al menos 1 carácter')
  .max(200, 'La búsqueda no puede exceder 200 caracteres')
  .trim();

export const validateFile = (file: File, maxSizeMB: number = 10): string | null => {
  const maxSize = maxSizeMB * 1024 * 1024;
  
  if (file.size > maxSize) {
    return `El archivo excede el tamaño máximo de ${maxSizeMB}MB`;
  }
  
  const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
  if (!validTypes.includes(file.type)) {
    return 'Tipo de archivo no válido. Solo se permiten imágenes (JPEG, PNG, WebP, GIF)';
  }
  
  return null;
};

export const validateFiles = (files: File[], maxFiles: number, maxSizeMB: number = 10): string | null => {
  if (files.length > maxFiles) {
    return `Máximo ${maxFiles} archivo${maxFiles > 1 ? 's' : ''} permitido${maxFiles > 1 ? 's' : ''}`;
  }
  
  for (const file of files) {
    const error = validateFile(file, maxSizeMB);
    if (error) {
      return error;
    }
  }
  
  return null;
};

