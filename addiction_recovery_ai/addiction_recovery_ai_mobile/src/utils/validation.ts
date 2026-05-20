import { z } from 'zod';

// Validation schemas using Zod
export const loginSchema = z.object({
  user_id: z.string().min(1, 'El ID de usuario es requerido'),
  password: z.string().optional(),
});

export const registerSchema = z.object({
  user_id: z.string().min(1, 'El ID de usuario es requerido'),
  email: z.string().email('Email inválido').optional().or(z.literal('')),
  name: z.string().optional(),
  password: z.string().min(8, 'La contraseña debe tener al menos 8 caracteres').optional().or(z.literal('')),
});

export const assessmentSchema = z.object({
  addiction_type: z.enum(['smoking', 'alcohol', 'drugs', 'gambling', 'internet', 'other'], {
    errorMap: () => ({ message: 'Tipo de adicción inválido' }),
  }),
  severity: z.enum(['low', 'moderate', 'high', 'severe'], {
    errorMap: () => ({ message: 'Nivel de severidad inválido' }),
  }),
  frequency: z.enum(['daily', 'weekly', 'monthly', 'occasional'], {
    errorMap: () => ({ message: 'Frecuencia inválida' }),
  }),
  duration_years: z.number().min(0).optional().or(z.literal('')),
  daily_cost: z.number().min(0).optional().or(z.literal('')),
  previous_attempts: z.number().int().min(0).default(0),
  support_system: z.boolean().default(false),
  triggers: z.array(z.string()).default([]),
  motivations: z.array(z.string()).default([]),
  medical_conditions: z.array(z.string()).default([]),
  additional_info: z.string().optional(),
});

export const logEntrySchema = z.object({
  user_id: z.string().min(1, 'ID de usuario requerido'),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'Formato de fecha inválido (YYYY-MM-DD)'),
  mood: z.enum(['excellent', 'good', 'neutral', 'poor', 'terrible'], {
    errorMap: () => ({ message: 'Estado de ánimo inválido' }),
  }),
  cravings_level: z.number().int().min(0).max(10, 'El nivel de ansias debe estar entre 0 y 10'),
  triggers_encountered: z.array(z.string()).default([]),
  consumed: z.boolean().default(false),
  notes: z.string().optional(),
});

// Type inference from schemas
export type LoginFormData = z.infer<typeof loginSchema>;
export type RegisterFormData = z.infer<typeof registerSchema>;
export type AssessmentFormData = z.infer<typeof assessmentSchema>;
export type LogEntryFormData = z.infer<typeof logEntrySchema>;

