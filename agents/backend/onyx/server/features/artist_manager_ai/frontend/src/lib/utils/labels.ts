import { EventType, RoutineType, ProtocolPriority, DressCode } from '@/types';

export const getEventTypeLabel = (type: EventType): string => {
  const labels: Record<EventType, string> = {
    [EventType.CONCERT]: 'Concierto',
    [EventType.INTERVIEW]: 'Entrevista',
    [EventType.PHOTOSHOOT]: 'Sesión de Fotos',
    [EventType.MEETING]: 'Reunión',
    [EventType.REHEARSAL]: 'Ensayo',
    [EventType.TRAVEL]: 'Viaje',
    [EventType.OTHER]: 'Otro',
  };
  return labels[type] || type;
};

export const getRoutineTypeLabel = (type: RoutineType): string => {
  const labels: Record<RoutineType, string> = {
    [RoutineType.MORNING]: 'Mañana',
    [RoutineType.AFTERNOON]: 'Tarde',
    [RoutineType.EVENING]: 'Noche',
    [RoutineType.NIGHT]: 'Noche',
    [RoutineType.CUSTOM]: 'Personalizado',
  };
  return labels[type] || type;
};

export const getProtocolPriorityLabel = (priority: ProtocolPriority): string => {
  const labels: Record<ProtocolPriority, string> = {
    [ProtocolPriority.CRITICAL]: 'Crítico',
    [ProtocolPriority.HIGH]: 'Alto',
    [ProtocolPriority.MEDIUM]: 'Medio',
    [ProtocolPriority.LOW]: 'Bajo',
  };
  return labels[priority] || priority;
};

export const getDressCodeLabel = (dressCode: DressCode): string => {
  const labels: Record<DressCode, string> = {
    [DressCode.FORMAL]: 'Formal',
    [DressCode.SMART_CASUAL]: 'Smart Casual',
    [DressCode.CASUAL]: 'Casual',
    [DressCode.SPORTY]: 'Deportivo',
    [DressCode.ELEGANT]: 'Elegante',
    [DressCode.STREETWEAR]: 'Streetwear',
  };
  return labels[dressCode] || dressCode;
};

