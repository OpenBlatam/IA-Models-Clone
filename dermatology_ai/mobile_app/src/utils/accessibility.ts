/**
 * Accessibility utilities
 */

export const accessibilityProps = {
  // Common accessibility labels
  labels: {
    close: 'Cerrar',
    back: 'Atrás',
    next: 'Siguiente',
    save: 'Guardar',
    delete: 'Eliminar',
    edit: 'Editar',
    share: 'Compartir',
    search: 'Buscar',
    filter: 'Filtrar',
    camera: 'Cámara',
    gallery: 'Galería',
    analysis: 'Análisis',
    recommendations: 'Recomendaciones',
    history: 'Historial',
    profile: 'Perfil',
  },

  // Helper function to create accessibility props
  createProps: (label: string, hint?: string) => ({
    accessible: true,
    accessibilityLabel: label,
    accessibilityHint: hint,
    accessibilityRole: 'button' as const,
  }),

  // Screen reader announcements
  announce: (message: string) => {
    // This would typically use a screen reader API
    // For now, it's a placeholder
    console.log('Screen reader:', message);
  },
};

