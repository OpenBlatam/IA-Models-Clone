/**
 * Navigation configuration constants
 */

export const NAVIGATION_CONFIG = {
  TAB: {
    ACTIVE_COLOR: '#6366f1',
    INACTIVE_COLOR: 'gray',
  },
  HEADER: {
    BACKGROUND_COLOR: '#6366f1',
    TINT_COLOR: '#fff',
    TITLE_STYLE: {
      fontWeight: 'bold' as const,
    },
  },
  SCREENS: {
    MAIN_TABS: {
      name: 'MainTabs',
      options: { headerShown: false },
    },
    REAL_TIME_SCAN: {
      name: 'RealTimeScan',
      title: 'Escaneo en Tiempo Real',
    },
    ANALYSIS: {
      name: 'Analysis',
      title: 'Análisis de Piel',
    },
    RECOMMENDATIONS: {
      name: 'Recommendations',
      title: 'Recomendaciones',
    },
    REPORT: {
      name: 'Report',
      title: 'Reporte Detallado',
    },
    COMPARISON: {
      name: 'Comparison',
      title: 'Comparar Análisis',
    },
  },
} as const;

export const TAB_ICONS = {
  Home: {
    focused: 'home' as const,
    unfocused: 'home-outline' as const,
  },
  Camera: {
    focused: 'camera' as const,
    unfocused: 'camera-outline' as const,
  },
  History: {
    focused: 'time' as const,
    unfocused: 'time-outline' as const,
  },
  Profile: {
    focused: 'person' as const,
    unfocused: 'person-outline' as const,
  },
} as const;

