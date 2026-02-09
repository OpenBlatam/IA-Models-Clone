/**
 * Constants for E2E tests
 * 
 * Centralized configuration for Playwright E2E tests
 */

/**
 * Test configuration
 */
export const TEST_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001',
  DEFAULT_TIMEOUT: 120000, // 2 minutes
  SHORT_TIMEOUT: 30000, // 30 seconds
  POLLING_INTERVAL: 1000, // 1 second
  MAX_POLLING_ATTEMPTS: 60, // 60 seconds max
  CONTENT_WAIT_THRESHOLD: 10, // seconds before warning
  MIN_CONTENT_LENGTH: 50, // minimum characters to consider content received
} as const;

/**
 * Page routes
 */
export const ROUTES = {
  AGENT_CONTROL: '/agent-control',
  API_TASKS: '/api/tasks',
} as const;

/**
 * Selectors
 */
export const SELECTORS = {
  INSTRUCTION_TEXTAREA: 'textarea[name="instruction"]',
  CREATE_BUTTON: 'button',
  TASK_CARD: '[data-testid="task-card"]',
  TASK_CARD_FALLBACK: '.task-card',
  TASK_CARD_CLASS: '[class*="task"]',
  TASK_STATUS_ATTR: 'data-status',
} as const;

/**
 * Button text patterns
 */
export const BUTTON_PATTERNS = {
  CREATE_OR_PROCESS: /crear|procesar/i,
} as const;

/**
 * Task status values
 */
export const TASK_STATUS = {
  PENDING: 'pending',
  COMPLETED: 'completed',
  PENDING_APPROVAL: 'pending_approval',
  FAILED: 'failed',
} as const;

/**
 * Task status indicators in text
 */
export const TASK_STATUS_INDICATORS = {
  COMPLETED: ['completada', 'completed', 'éxito', 'success'],
  FAILED: ['error', 'falló', 'failed', 'error'],
} as const;

/**
 * Problematic log patterns
 */
export const PROBLEMATIC_LOG_PATTERNS = [
  'Stream se cerró prematuramente',
  'Stream se cerró',
  '0 caracteres',
  '{}',
] as const;

/**
 * Test instructions
 */
export const TEST_INSTRUCTIONS = {
  DEFAULT: 'Crea un archivo README.md con "Hello World"',
  SIMPLE: 'Test',
} as const;

/**
 * Error messages
 */
export const ERROR_MESSAGES = {
  TASK_FAILED: (details: string) => `Tarea falló: ${details}`,
  NO_CONTENT_RECEIVED: (seconds: number) => 
    `No se recibió contenido después de ${seconds} segundos`,
  PROBLEMATIC_LOGS_DETECTED: 'Se detectaron logs que indican problemas con el stream',
  TASK_NOT_FOUND: 'No se encontró ninguna tarea en la página',
  API_ERROR: (status: string, contentLength: number) => 
    `Error en API - Estado: ${status}, Contenido: ${contentLength} caracteres`,
} as const;

/**
 * Success messages
 */
export const SUCCESS_MESSAGES = {
  PAGE_LOADED: '✅ Página cargada',
  INSTRUCTION_ENTERED: '✅ Instrucción ingresada',
  BUTTON_CLICKED: '✅ Botón clickeado',
  TASK_CREATED: '✅ Tarea creada',
  CONTENT_RECEIVED: (length: number) => `✅ Contenido recibido: ${length} caracteres`,
  TASK_COMPLETED: (status: string) => `✅ Tarea completada: ${status}`,
  NO_PROBLEMATIC_LOGS: (totalLogs: number) => 
    `✅ No se encontraron logs problemáticos (${totalLogs} logs totales)`,
} as const;

/**
 * Log messages
 */
export const LOG_MESSAGES = {
  POLLING_UPDATE: (second: number, preview: string) => 
    `📊 Segundo ${second}: ${preview}`,
  PROBLEMATIC_LOG_DETECTED: (text: string) => 
    `❌ Log problemático detectado: ${text}`,
  PROBLEM_DETECTED: (seconds: number) => 
    `❌ PROBLEMA: Después de ${seconds} segundos, no hay contenido`,
  SCREENSHOT_SAVED: '❌ Screenshot guardado para análisis',
  API_STATUS: (status: string) => `❌ Estado API: ${status}`,
  API_CONTENT: (length: number) => `❌ Contenido API: ${length} caracteres`,
  PROBLEMATIC_LOGS_FOUND: '❌ Logs problemáticos encontrados:',
} as const;



