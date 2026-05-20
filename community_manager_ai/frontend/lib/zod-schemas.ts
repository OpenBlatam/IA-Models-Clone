/**
 * Zod Validation Schemas
 * Type-safe validation schemas for form inputs and API data
 */

import { z } from 'zod';
import { PLATFORMS, POST_STATUS } from '@/lib/config/constants';

/**
 * Validation constants
 */
const VALIDATION_LIMITS = {
  POST_CONTENT_MIN: 1,
  POST_CONTENT_MAX: 5000,
  MEME_CAPTION_MAX: 500,
  TEMPLATE_NAME_MIN: 1,
  TEMPLATE_NAME_MAX: 100,
  TEMPLATE_CONTENT_MIN: 1,
  TEMPLATE_CONTENT_MAX: 10000,
  CATEGORY_MAX: 100,
  TAG_MAX_LENGTH: 50,
  MAX_TAGS: 10,
} as const;

/**
 * Platform enum schema
 */
const platformEnum = z.enum([
  PLATFORMS.FACEBOOK,
  PLATFORMS.INSTAGRAM,
  PLATFORMS.TWITTER,
  PLATFORMS.LINKEDIN,
  PLATFORMS.TIKTOK,
  PLATFORMS.YOUTUBE,
]);

/**
 * Post status enum schema
 */
const postStatusEnum = z.enum([
  POST_STATUS.SCHEDULED,
  POST_STATUS.PUBLISHED,
  POST_STATUS.CANCELLED,
]);

/**
 * ISO date string schema
 */
const isoDateString = z.string().refine(
  (val) => {
    const date = new Date(val);
    return !isNaN(date.getTime());
  },
  { message: 'Fecha inválida' }
);

/**
 * URL schema
 */
const urlSchema = z.string().url().or(z.string().startsWith('/'));

/**
 * Post creation schema
 */
export const postSchema = z.object({
  content: z
    .string()
    .min(VALIDATION_LIMITS.POST_CONTENT_MIN, 'El contenido es requerido')
    .max(VALIDATION_LIMITS.POST_CONTENT_MAX, 'El contenido es demasiado largo'),
  platforms: z
    .array(platformEnum)
    .min(1, 'Selecciona al menos una plataforma')
    .max(6, 'Máximo 6 plataformas'),
  scheduled_time: isoDateString.optional(),
  media_paths: z.array(urlSchema).max(10, 'Máximo 10 archivos multimedia').optional(),
  tags: z
    .array(z.string().max(VALIDATION_LIMITS.TAG_MAX_LENGTH, 'El tag es demasiado largo'))
    .max(VALIDATION_LIMITS.MAX_TAGS, `Máximo ${VALIDATION_LIMITS.MAX_TAGS} tags`)
    .optional(),
});

/**
 * Post update schema (all fields optional)
 */
export const postUpdateSchema = postSchema.partial().refine(
  (data) => {
    // At least one field must be provided
    return Object.keys(data).length > 0;
  },
  { message: 'Al menos un campo debe ser actualizado' }
);

/**
 * Meme creation schema
 */
export const memeSchema = z.object({
  caption: z
    .string()
    .max(VALIDATION_LIMITS.MEME_CAPTION_MAX, 'El caption es demasiado largo')
    .optional(),
  tags: z
    .array(z.string().max(VALIDATION_LIMITS.TAG_MAX_LENGTH, 'El tag es demasiado largo'))
    .max(VALIDATION_LIMITS.MAX_TAGS, `Máximo ${VALIDATION_LIMITS.MAX_TAGS} tags`)
    .optional(),
  category: z
    .string()
    .max(VALIDATION_LIMITS.CATEGORY_MAX, 'La categoría es demasiado larga')
    .optional(),
});

/**
 * Template creation schema
 */
export const templateSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_LIMITS.TEMPLATE_NAME_MIN, 'El nombre es requerido')
    .max(VALIDATION_LIMITS.TEMPLATE_NAME_MAX, 'El nombre es demasiado largo')
    .trim(),
  content: z
    .string()
    .min(VALIDATION_LIMITS.TEMPLATE_CONTENT_MIN, 'El contenido es requerido')
    .max(VALIDATION_LIMITS.TEMPLATE_CONTENT_MAX, 'El contenido es demasiado largo'),
  variables: z
    .array(z.string().max(VALIDATION_LIMITS.TAG_MAX_LENGTH, 'El nombre de variable es demasiado largo'))
    .optional(),
  category: z
    .string()
    .max(VALIDATION_LIMITS.CATEGORY_MAX, 'La categoría es demasiado larga')
    .optional(),
});

/**
 * Platform connection schema
 */
export const platformConnectSchema = z.object({
  platform: platformEnum,
  credentials: z.record(z.string(), z.string()).refine(
    (credentials) => Object.keys(credentials).length > 0,
    { message: 'Las credenciales son requeridas' }
  ),
});

/**
 * Date range schema for analytics
 */
export const dateRangeSchema = z.object({
  startDate: isoDateString.optional(),
  endDate: isoDateString.optional(),
}).refine(
  (data) => {
    if (data.startDate && data.endDate) {
      return new Date(data.startDate) <= new Date(data.endDate);
    }
    return true;
  },
  { message: 'La fecha de inicio debe ser anterior a la fecha de fin' }
);

/**
 * Pagination schema
 */
export const paginationSchema = z.object({
  page: z.number().int().min(1).default(1),
  limit: z.number().int().min(1).max(100).default(10),
});

/**
 * Search/filter schema
 */
export const searchSchema = z.object({
  query: z.string().max(200, 'La búsqueda es demasiado larga').optional(),
  category: z.string().max(VALIDATION_LIMITS.CATEGORY_MAX).optional(),
  tags: z.array(z.string()).optional(),
  status: postStatusEnum.optional(),
  platform: platformEnum.optional(),
});

// Type exports
export type PostFormData = z.infer<typeof postSchema>;
export type PostUpdateFormData = z.infer<typeof postUpdateSchema>;
export type MemeFormData = z.infer<typeof memeSchema>;
export type TemplateFormData = z.infer<typeof templateSchema>;
export type PlatformConnectFormData = z.infer<typeof platformConnectSchema>;
export type DateRangeFormData = z.infer<typeof dateRangeSchema>;
export type PaginationFormData = z.infer<typeof paginationSchema>;
export type SearchFormData = z.infer<typeof searchSchema>;


