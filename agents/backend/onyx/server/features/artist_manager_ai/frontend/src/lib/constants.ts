import { EventType, RoutineType, ProtocolCategory, ProtocolPriority, DressCode, Season } from '@/types';

export const EVENT_TYPE_OPTIONS = [
  { value: EventType.CONCERT, label: 'Concierto' },
  { value: EventType.INTERVIEW, label: 'Entrevista' },
  { value: EventType.PHOTOSHOOT, label: 'Sesión de Fotos' },
  { value: EventType.MEETING, label: 'Reunión' },
  { value: EventType.REHEARSAL, label: 'Ensayo' },
  { value: EventType.TRAVEL, label: 'Viaje' },
  { value: EventType.OTHER, label: 'Otro' },
];

export const ROUTINE_TYPE_OPTIONS = [
  { value: RoutineType.MORNING, label: 'Mañana' },
  { value: RoutineType.AFTERNOON, label: 'Tarde' },
  { value: RoutineType.EVENING, label: 'Noche' },
  { value: RoutineType.NIGHT, label: 'Noche' },
  { value: RoutineType.CUSTOM, label: 'Personalizado' },
];

export const DAY_OPTIONS = [
  { value: 0, label: 'Domingo' },
  { value: 1, label: 'Lunes' },
  { value: 2, label: 'Martes' },
  { value: 3, label: 'Miércoles' },
  { value: 4, label: 'Jueves' },
  { value: 5, label: 'Viernes' },
  { value: 6, label: 'Sábado' },
];

export const PROTOCOL_CATEGORY_OPTIONS = [
  { value: ProtocolCategory.SOCIAL_MEDIA, label: 'Redes Sociales' },
  { value: ProtocolCategory.INTERVIEW, label: 'Entrevista' },
  { value: ProtocolCategory.CONCERT, label: 'Concierto' },
  { value: ProtocolCategory.PHOTOSHOOT, label: 'Sesión de Fotos' },
  { value: ProtocolCategory.TRAVEL, label: 'Viaje' },
  { value: ProtocolCategory.GENERAL, label: 'General' },
];

export const PROTOCOL_PRIORITY_OPTIONS = [
  { value: ProtocolPriority.CRITICAL, label: 'Crítico' },
  { value: ProtocolPriority.HIGH, label: 'Alto' },
  { value: ProtocolPriority.MEDIUM, label: 'Medio' },
  { value: ProtocolPriority.LOW, label: 'Bajo' },
];

export const SEASON_OPTIONS = [
  { value: Season.SPRING, label: 'Primavera' },
  { value: Season.SUMMER, label: 'Verano' },
  { value: Season.FALL, label: 'Otoño' },
  { value: Season.WINTER, label: 'Invierno' },
  { value: Season.ALL_SEASON, label: 'Todas las Estaciones' },
];

export const DRESS_CODE_OPTIONS = [
  { value: DressCode.FORMAL, label: 'Formal' },
  { value: DressCode.SMART_CASUAL, label: 'Smart Casual' },
  { value: DressCode.CASUAL, label: 'Casual' },
  { value: DressCode.SPORTY, label: 'Deportivo' },
  { value: DressCode.ELEGANT, label: 'Elegante' },
  { value: DressCode.STREETWEAR, label: 'Streetwear' },
];

