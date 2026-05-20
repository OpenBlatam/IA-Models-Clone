/**
 * Validation Utilities
 * ====================
 * Zod schemas for form validation
 */

import { z } from 'zod';

export const manualDescriptionSchema = z
  .string()
  .min(10, 'La descripción debe tener al menos 10 caracteres')
  .max(2000, 'La descripción no puede exceder 2000 caracteres');

export const categorySchema = z.enum([
  'plomeria',
  'techos',
  'carpinteria',
  'electricidad',
  'albanileria',
  'pintura',
  'herreria',
  'jardineria',
  'general',
]);

export const manualTextRequestSchema = z.object({
  problem_description: manualDescriptionSchema,
  category: categorySchema.optional(),
  model: z.string().optional(),
  include_safety: z.boolean().default(true),
  include_tools: z.boolean().default(true),
  include_materials: z.boolean().default(true),
});

export type ManualTextRequestForm = z.infer<typeof manualTextRequestSchema>;




