/**
 * Accessibility utilities
 * @module robot-3d-view/utils/accessibility
 */

/**
 * ARIA labels for 3D view components
 */
export const ariaLabels = {
  view3D: 'Visualización 3D del robot',
  loading: 'Cargando vista 3D',
  controls: 'Controles de visualización',
  info: 'Información del robot',
  instructions: 'Instrucciones de uso',
  screenshot: 'Capturar screenshot',
  toggleStats: 'Alternar estadísticas de rendimiento',
  toggleGizmo: 'Alternar gizmo de navegación',
  toggleStars: 'Alternar fondo estrellado',
  toggleWaypoints: 'Alternar puntos de ruta',
  toggleGrid: 'Alternar grid',
  toggleObjects: 'Alternar objetos de entorno',
  toggleAutoRotate: 'Alternar rotación automática',
  cameraPreset: (preset: string) => `Cambiar a vista ${preset}`,
} as const;

/**
 * Keyboard shortcuts configuration
 */
export const keyboardShortcuts = {
  toggleStats: 'KeyS',
  toggleGizmo: 'KeyG',
  toggleGrid: 'KeyH',
  toggleObjects: 'KeyO',
  toggleAutoRotate: 'KeyR',
  screenshot: 'KeyP',
  cameraFront: 'Digit1',
  cameraTop: 'Digit2',
  cameraSide: 'Digit3',
  cameraIso: 'Digit4',
} as const;

/**
 * Gets ARIA label for a component
 * 
 * @param key - Label key
 * @param params - Optional parameters for label interpolation
 * @returns ARIA label string
 */
export function getAriaLabel(
  key: keyof typeof ariaLabels,
  params?: Record<string, string>
): string {
  const label = ariaLabels[key];
  if (typeof label === 'function' && params) {
    return label(params.preset || '');
  }
  return typeof label === 'string' ? label : '';
}

/**
 * Gets keyboard shortcut description
 * 
 * @param key - Shortcut key
 * @returns Human-readable shortcut description
 */
export function getKeyboardShortcut(key: keyof typeof keyboardShortcuts): string {
  const shortcuts: Record<string, string> = {
    KeyS: 'S',
    KeyG: 'G',
    KeyH: 'H',
    KeyO: 'O',
    KeyR: 'R',
    KeyP: 'P',
    Digit1: '1',
    Digit2: '2',
    Digit3: '3',
    Digit4: '4',
  };
  return shortcuts[key] || '';
}



