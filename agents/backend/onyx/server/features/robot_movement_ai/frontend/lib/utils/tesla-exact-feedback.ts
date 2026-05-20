/**
 * Tesla Exact Feedback System
 * Exact colors, spacing, and styles for feedback components
 */

export const teslaExactFeedback = {
  // Success Colors (exact)
  success: {
    background: '#d1fae5',
    border: '#10b981',
    text: '#065f46',
    icon: '#10b981',
    hover: '#10b981',
  },

  // Error Colors (exact)
  error: {
    background: '#fee2e2',
    border: '#ef4444',
    text: '#991b1b',
    icon: '#ef4444',
    hover: '#ef4444',
  },

  // Warning Colors (exact)
  warning: {
    background: '#fef3c7',
    border: '#f59e0b',
    text: '#92400e',
    icon: '#f59e0b',
    hover: '#f59e0b',
  },

  // Info Colors (exact)
  info: {
    background: '#dbeafe',
    border: '#3b82f6',
    text: '#1e40af',
    icon: '#3b82f6',
    hover: '#3b82f6',
  },

  // Status Colors (exact)
  status: {
    online: {
      color: '#10b981',
      pulse: 'rgba(16, 185, 129, 0.3)',
    },
    offline: {
      color: '#9ca3af',
      pulse: 'rgba(156, 163, 175, 0.3)',
    },
    away: {
      color: '#f59e0b',
      pulse: 'rgba(245, 158, 11, 0.3)',
    },
    busy: {
      color: '#ef4444',
      pulse: 'rgba(239, 68, 68, 0.3)',
    },
    loading: {
      color: '#0062cc',
      pulse: 'rgba(0, 98, 204, 0.3)',
    },
  },

  // Progress Colors (exact)
  progress: {
    blue: '#0062cc',
    green: '#10b981',
    yellow: '#f59e0b',
    red: '#ef4444',
    track: '#e5e7eb',
  },

  // Spacing (exact)
  spacing: {
    padding: '16px', // p-4
    gap: '12px', // gap-3
    iconSize: '20px', // w-5 h-5
    borderRadius: '8px', // rounded-lg
  },

  // Typography (exact)
  typography: {
    title: {
      fontSize: '14px',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    message: {
      fontSize: '14px',
      fontWeight: 400,
      lineHeight: 1.5,
    },
  },

  // Animation (exact)
  animation: {
    duration: 200, // ms
    easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
    pulseDuration: 2000, // ms
  },
} as const;

// Helper functions
export function getTeslaFeedbackColor(
  type: 'success' | 'error' | 'warning' | 'info',
  property: 'background' | 'border' | 'text' | 'icon' | 'hover'
): string {
  return teslaExactFeedback[type][property];
}

export function getTeslaStatusColor(
  status: 'online' | 'offline' | 'away' | 'busy' | 'loading',
  property: 'color' | 'pulse'
): string {
  return teslaExactFeedback.status[status][property];
}

export function getTeslaProgressColor(color: 'blue' | 'green' | 'yellow' | 'red'): string {
  return teslaExactFeedback.progress[color];
}



