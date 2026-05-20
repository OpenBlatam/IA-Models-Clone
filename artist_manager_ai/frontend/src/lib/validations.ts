import { z } from 'zod';
import { EventType, RoutineType, ProtocolCategory, ProtocolPriority, DressCode, Season } from '@/types';

export const eventSchema = z.object({
  title: z.string().min(1, 'El título es requerido').max(200, 'El título es demasiado largo'),
  description: z.string().min(1, 'La descripción es requerida'),
  event_type: z.nativeEnum(EventType),
  start_time: z.string().min(1, 'La fecha de inicio es requerida'),
  end_time: z.string().min(1, 'La fecha de fin es requerida'),
  location: z.string().optional(),
  attendees: z.array(z.string()).optional(),
  protocol_requirements: z.array(z.string()).optional(),
  wardrobe_requirements: z.string().optional(),
  notes: z.string().optional(),
}).refine((data) => {
  if (data.start_time && data.end_time) {
    return new Date(data.start_time) < new Date(data.end_time);
  }
  return true;
}, {
  message: 'La fecha de fin debe ser posterior a la fecha de inicio',
  path: ['end_time'],
});

export const routineSchema = z.object({
  title: z.string().min(1, 'El título es requerido').max(200, 'El título es demasiado largo'),
  description: z.string().min(1, 'La descripción es requerida'),
  routine_type: z.nativeEnum(RoutineType),
  scheduled_time: z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Formato de hora inválido (HH:MM)'),
  duration_minutes: z.number().min(1, 'La duración debe ser al menos 1 minuto').max(1440, 'La duración no puede exceder 24 horas'),
  priority: z.number().min(1, 'La prioridad debe ser al menos 1').max(10, 'La prioridad no puede exceder 10'),
  days_of_week: z.array(z.number().min(0).max(6)).min(1, 'Debe seleccionar al menos un día'),
  is_required: z.boolean().optional(),
  notes: z.string().optional(),
});

export const protocolSchema = z.object({
  title: z.string().min(1, 'El título es requerido').max(200, 'El título es demasiado largo'),
  description: z.string().min(1, 'La descripción es requerida'),
  category: z.nativeEnum(ProtocolCategory),
  priority: z.nativeEnum(ProtocolPriority),
  rules: z.array(z.string().min(1, 'Las reglas no pueden estar vacías')).min(1, 'Debe agregar al menos una regla'),
  do_s: z.array(z.string()).optional(),
  dont_s: z.array(z.string()).optional(),
  context: z.string().optional(),
  applicable_events: z.array(z.string()).optional(),
  notes: z.string().optional(),
});

export const wardrobeItemSchema = z.object({
  name: z.string().min(1, 'El nombre es requerido').max(200, 'El nombre es demasiado largo'),
  category: z.string().min(1, 'La categoría es requerida'),
  color: z.string().min(1, 'El color es requerido'),
  brand: z.string().optional(),
  size: z.string().optional(),
  season: z.nativeEnum(Season),
  dress_codes: z.array(z.nativeEnum(DressCode)).min(1, 'Debe seleccionar al menos un código de vestimenta'),
  notes: z.string().optional(),
  image_url: z.string().url('URL inválida').optional().or(z.literal('')),
});

export const outfitSchema = z.object({
  name: z.string().min(1, 'El nombre es requerido').max(200, 'El nombre es demasiado largo'),
  items: z.array(z.string()).min(1, 'Debe seleccionar al menos un item'),
  dress_code: z.nativeEnum(DressCode),
  occasion: z.string().min(1, 'La ocasión es requerida'),
  season: z.nativeEnum(Season),
  notes: z.string().optional(),
  image_url: z.string().url('URL inválida').optional().or(z.literal('')),
});

export type EventFormData = z.infer<typeof eventSchema>;
export type RoutineFormData = z.infer<typeof routineSchema>;
export type ProtocolFormData = z.infer<typeof protocolSchema>;
export type WardrobeItemFormData = z.infer<typeof wardrobeItemSchema>;
export type OutfitFormData = z.infer<typeof outfitSchema>;

