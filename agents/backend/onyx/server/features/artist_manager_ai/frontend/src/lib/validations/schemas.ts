import { z } from 'zod';
import { EventType, RoutineType, ProtocolCategory, ProtocolPriority, DressCode, Season } from '@/types';
import { VALIDATION_RULES, ERROR_MESSAGES } from '@/lib/constants/validation';

export const eventSchema = z
  .object({
    title: z
      .string()
      .min(VALIDATION_RULES.title.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.title.minLength))
      .max(VALIDATION_RULES.title.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.title.maxLength)),
    description: z
      .string()
      .min(VALIDATION_RULES.description.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.description.minLength))
      .max(VALIDATION_RULES.description.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.description.maxLength)),
    event_type: z.nativeEnum(EventType),
    start_time: z.string().min(1, ERROR_MESSAGES.required),
    end_time: z.string().min(1, ERROR_MESSAGES.required),
    location: z.string().optional(),
    attendees: z.array(z.string()).default([]),
    protocol_requirements: z.array(z.string()).default([]),
    wardrobe_requirements: z.string().optional(),
    notes: z.string().optional(),
  })
  .refine(
    (data) => {
      if (data.start_time && data.end_time) {
        return new Date(data.start_time) < new Date(data.end_time);
      }
      return true;
    },
    {
      message: 'La fecha de fin debe ser posterior a la fecha de inicio',
      path: ['end_time'],
    }
  );

export const routineSchema = z.object({
  title: z
    .string()
    .min(VALIDATION_RULES.title.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.title.minLength))
    .max(VALIDATION_RULES.title.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.title.maxLength)),
  description: z
    .string()
    .min(VALIDATION_RULES.description.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.description.minLength))
    .max(VALIDATION_RULES.description.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.description.maxLength)),
  routine_type: z.nativeEnum(RoutineType),
  scheduled_time: z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/, ERROR_MESSAGES.invalidTime),
  duration_minutes: z
    .number()
    .min(VALIDATION_RULES.duration.min, ERROR_MESSAGES.min(VALIDATION_RULES.duration.min))
    .max(VALIDATION_RULES.duration.max, ERROR_MESSAGES.max(VALIDATION_RULES.duration.max)),
  priority: z
    .number()
    .min(VALIDATION_RULES.priority.min, ERROR_MESSAGES.min(VALIDATION_RULES.priority.min))
    .max(VALIDATION_RULES.priority.max, ERROR_MESSAGES.max(VALIDATION_RULES.priority.max)),
  days_of_week: z.array(z.number().min(0).max(6)).min(1, 'Debe seleccionar al menos un día'),
  is_required: z.boolean().default(true),
  notes: z.string().optional(),
});

export const protocolSchema = z.object({
  title: z
    .string()
    .min(VALIDATION_RULES.title.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.title.minLength))
    .max(VALIDATION_RULES.title.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.title.maxLength)),
  description: z
    .string()
    .min(VALIDATION_RULES.description.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.description.minLength))
    .max(VALIDATION_RULES.description.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.description.maxLength)),
  category: z.nativeEnum(ProtocolCategory),
  priority: z.nativeEnum(ProtocolPriority),
  rules: z.array(z.string().min(1, 'La regla no puede estar vacía')).min(1, 'Debe tener al menos una regla'),
  do_s: z.array(z.string()).default([]),
  dont_s: z.array(z.string()).default([]),
  context: z.string().optional(),
  notes: z.string().optional(),
});

export const wardrobeItemSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_RULES.title.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.title.minLength))
    .max(VALIDATION_RULES.title.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.title.maxLength)),
  category: z.string().min(1, ERROR_MESSAGES.required),
  color: z.string().min(1, ERROR_MESSAGES.required),
  brand: z.string().optional(),
  size: z.string().optional(),
  season: z.nativeEnum(Season),
  dress_codes: z.array(z.nativeEnum(DressCode)).min(1, 'Debe seleccionar al menos un código de vestimenta'),
  image_url: z.string().url(ERROR_MESSAGES.url).optional().or(z.literal('')),
  notes: z.string().optional(),
});

export const outfitSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_RULES.title.minLength, ERROR_MESSAGES.minLength(VALIDATION_RULES.title.minLength))
    .max(VALIDATION_RULES.title.maxLength, ERROR_MESSAGES.maxLength(VALIDATION_RULES.title.maxLength)),
  items: z.array(z.string()).min(1, 'Debe seleccionar al menos un item'),
  dress_code: z.nativeEnum(DressCode),
  occasion: z.string().min(1, ERROR_MESSAGES.required),
  season: z.nativeEnum(Season),
  notes: z.string().optional(),
  image_url: z.string().url(ERROR_MESSAGES.url).optional().or(z.literal('')),
});

export type EventFormData = z.infer<typeof eventSchema>;
export type RoutineFormData = z.infer<typeof routineSchema>;
export type ProtocolFormData = z.infer<typeof protocolSchema>;
export type WardrobeItemFormData = z.infer<typeof wardrobeItemSchema>;
export type OutfitFormData = z.infer<typeof outfitSchema>;

