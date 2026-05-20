/**
 * Theme system for Robot 3D View
 * 
 * Provides theme configurations for different visual styles.
 * 
 * @module robot-3d-view/lib/themes
 */

/**
 * Theme type
 */
export type ThemeType = 'dark' | 'light' | 'industrial' | 'futuristic' | 'minimal';

/**
 * Color palette for themes
 */
export interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  grid: {
    cell: string;
    section: string;
  };
  robot: {
    base: string;
    link: string;
    joint: string;
    effector: string;
    gripper: string;
  };
  target: string;
  trajectory: string;
}

/**
 * Theme configurations
 */
export const THEMES: Record<ThemeType, ThemeColors> = {
  dark: {
    primary: '#0ea5e9',
    secondary: '#0284c7',
    accent: '#10b981',
    background: '#0f172a',
    surface: '#1e293b',
    text: '#f1f5f9',
    grid: {
      cell: '#374151',
      section: '#1f2937',
    },
    robot: {
      base: '#0ea5e9',
      link: '#0284c7',
      joint: '#0369a1',
      effector: '#10b981',
      gripper: '#059669',
    },
    target: '#f59e0b',
    trajectory: '#f59e0b',
  },
  light: {
    primary: '#3b82f6',
    secondary: '#2563eb',
    accent: '#10b981',
    background: '#ffffff',
    surface: '#f8fafc',
    text: '#1e293b',
    grid: {
      cell: '#e2e8f0',
      section: '#cbd5e1',
    },
    robot: {
      base: '#3b82f6',
      link: '#2563eb',
      joint: '#1d4ed8',
      effector: '#10b981',
      gripper: '#059669',
    },
    target: '#f59e0b',
    trajectory: '#f59e0b',
  },
  industrial: {
    primary: '#ef4444',
    secondary: '#dc2626',
    accent: '#f59e0b',
    background: '#1f2937',
    surface: '#374151',
    text: '#f9fafb',
    grid: {
      cell: '#4b5563',
      section: '#374151',
    },
    robot: {
      base: '#ef4444',
      link: '#dc2626',
      joint: '#b91c1c',
      effector: '#f59e0b',
      gripper: '#d97706',
    },
    target: '#fbbf24',
    trajectory: '#f59e0b',
  },
  futuristic: {
    primary: '#8b5cf6',
    secondary: '#7c3aed',
    accent: '#06b6d4',
    background: '#0a0a0f',
    surface: '#1a1a2e',
    text: '#e0e7ff',
    grid: {
      cell: '#312e81',
      section: '#1e1b4b',
    },
    robot: {
      base: '#8b5cf6',
      link: '#7c3aed',
      joint: '#6d28d9',
      effector: '#06b6d4',
      gripper: '#0891b2',
    },
    target: '#f59e0b',
    trajectory: '#06b6d4',
  },
  minimal: {
    primary: '#64748b',
    secondary: '#475569',
    accent: '#64748b',
    background: '#ffffff',
    surface: '#f1f5f9',
    text: '#0f172a',
    grid: {
      cell: '#e2e8f0',
      section: '#cbd5e1',
    },
    robot: {
      base: '#64748b',
      link: '#475569',
      joint: '#334155',
      effector: '#64748b',
      gripper: '#475569',
    },
    target: '#f59e0b',
    trajectory: '#64748b',
  },
};

/**
 * Gets theme colors
 * 
 * @param theme - Theme type
 * @returns Theme colors
 */
export function getTheme(theme: ThemeType): ThemeColors {
  return THEMES[theme];
}

/**
 * Applies theme to configuration
 * 
 * @param theme - Theme to apply
 * @returns Theme colors
 */
export function applyTheme(theme: ThemeType): ThemeColors {
  return getTheme(theme);
}



