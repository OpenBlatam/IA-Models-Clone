// Internationalization translations

export type Language = 'es' | 'en';

export interface Translations {
  [key: string]: {
    [key in Language]: string;
  };
}

export const translations: Translations = {
  'app.title': {
    es: 'Robot Movement AI',
    en: 'Robot Movement AI',
  },
  'app.subtitle': {
    es: 'Plataforma IA de Movimiento Robótico',
    en: 'AI Robotic Movement Platform',
  },
  'tabs.control': {
    es: 'Control',
    en: 'Control',
  },
  'tabs.chat': {
    es: 'Chat',
    en: 'Chat',
  },
  'tabs.status': {
    es: 'Estado',
    en: 'Status',
  },
  'tabs.metrics': {
    es: 'Métricas',
    en: 'Metrics',
  },
  'tabs.3d': {
    es: 'Vista 3D',
    en: '3D View',
  },
  'tabs.history': {
    es: 'Historial',
    en: 'History',
  },
  'tabs.optimize': {
    es: 'Optimizar',
    en: 'Optimize',
  },
  'tabs.recording': {
    es: 'Grabación',
    en: 'Recording',
  },
  'tabs.advanced': {
    es: 'Métricas Avanzadas',
    en: 'Advanced Metrics',
  },
  'tabs.compare': {
    es: 'Comparar',
    en: 'Compare',
  },
  'tabs.commands': {
    es: 'Comandos',
    en: 'Commands',
  },
  'tabs.widgets': {
    es: 'Widgets',
    en: 'Widgets',
  },
  'tabs.reports': {
    es: 'Reportes',
    en: 'Reports',
  },
  'tabs.predictive': {
    es: 'Predictivo',
    en: 'Predictive',
  },
  'tabs.logs': {
    es: 'Logs',
    en: 'Logs',
  },
  'tabs.alerts': {
    es: 'Alertas',
    en: 'Alerts',
  },
  'tabs.help': {
    es: 'Ayuda',
    en: 'Help',
  },
  'tabs.settings': {
    es: 'Configuración',
    en: 'Settings',
  },
  'tabs.collaboration': {
    es: 'Colaboración',
    en: 'Collaboration',
  },
  'common.connected': {
    es: 'Conectado',
    en: 'Connected',
  },
  'common.disconnected': {
    es: 'Desconectado',
    en: 'Disconnected',
  },
  'common.save': {
    es: 'Guardar',
    en: 'Save',
  },
  'common.cancel': {
    es: 'Cancelar',
    en: 'Cancel',
  },
  'common.delete': {
    es: 'Eliminar',
    en: 'Delete',
  },
  'common.edit': {
    es: 'Editar',
    en: 'Edit',
  },
  'common.close': {
    es: 'Cerrar',
    en: 'Close',
  },
  'common.search': {
    es: 'Buscar...',
    en: 'Search...',
  },
  'common.loading': {
    es: 'Cargando...',
    en: 'Loading...',
  },
  'common.error': {
    es: 'Error',
    en: 'Error',
  },
  'common.success': {
    es: 'Éxito',
    en: 'Success',
  },
};

export function getTranslation(key: string, language: Language): string {
  return translations[key]?.[language] || key;
}

