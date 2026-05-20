export const MESSAGES = {
  MANUAL: {
    GENERATE_SUCCESS: 'Manual generado exitosamente',
    GENERATE_ERROR: 'Error al generar el manual',
    LOADING: 'Cargando manual...',
    LOAD_ERROR: 'Error al cargar el manual',
    INVALID_ID: 'ID de manual inválido',
    NO_IMAGES: 'Debe seleccionar al menos una imagen',
    DESCRIPTION_REQUIRED: 'La descripción del problema es requerida',
  },
  RATING: {
    ADD_SUCCESS: 'Calificación agregada exitosamente',
    ADD_ERROR: 'Error al agregar calificación',
    SELECT_REQUIRED: 'Por favor selecciona una calificación',
  },
  FAVORITE: {
    ADD_SUCCESS: 'Agregado a favoritos',
    REMOVE_SUCCESS: 'Removido de favoritos',
    UPDATE_ERROR: 'Error al actualizar favoritos',
  },
  EXPORT: {
    SUCCESS: (format: string) => `Manual exportado como ${format.toUpperCase()}`,
    ERROR: 'Error al exportar manual',
  },
  SEARCH: {
    NO_RESULTS: 'No se encontraron resultados',
    NO_QUERY: 'Ingresa un término de búsqueda',
  },
  FILE: {
    MAX_FILES: (max: number, multiple: boolean) =>
      `Máximo ${max} ${multiple ? 'imágenes' : 'imagen'} permitida${multiple ? 's' : ''}`,
    MAX_SIZE: (size: number) => `Algunas imágenes exceden el tamaño máximo de ${size}MB`,
    INVALID_TYPE: 'Tipo de archivo no válido. Solo se permiten imágenes (JPEG, PNG, WebP, GIF)',
    INVALID_COUNT: 'Debe proporcionar entre 1 y 5 imágenes',
  },
  ANALYTICS: {
    LOADING: 'Cargando estadísticas...',
    LOAD_ERROR: 'Error al cargar las estadísticas',
    NO_CATEGORIES: 'No hay estadísticas de categorías',
    NO_CATEGORIES_DESC: 'No se encontraron datos de categorías para el período seleccionado',
  },
  EMPTY: {
    NO_MANUALS: 'No se encontraron manuales',
    NO_FAVORITES: 'No tienes manuales favoritos aún',
    NO_RATINGS: 'No hay calificaciones',
    NO_RATINGS_DESC: 'Sé el primero en calificar este manual',
  },
} as const;

