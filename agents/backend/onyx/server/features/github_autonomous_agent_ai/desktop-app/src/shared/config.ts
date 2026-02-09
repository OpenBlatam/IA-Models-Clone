/**
 * Configuración compartida entre main y renderer
 */

export const APP_CONFIG = {
  name: 'GitHub Autonomous Agent AI',
  version: '1.0.0',
  apiBaseUrl: process.env.API_URL || 'http://localhost:8030',
  wsUrl: process.env.WS_URL || 'ws://localhost:8030/ws',
} as const;


