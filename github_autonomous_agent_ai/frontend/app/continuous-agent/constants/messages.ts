/**
 * User-facing messages for Continuous Agent feature
 */

export const SUCCESS_MESSAGES = {
  AGENT_CREATED: "Agente creado exitosamente",
  AGENT_UPDATED: "Agente actualizado exitosamente",
  AGENT_DELETED: "Agente eliminado exitosamente",
  AGENT_ACTIVATED: "Agente activado",
  AGENT_DEACTIVATED: "Agente desactivado",
} as const;

export const ERROR_MESSAGES = {
  CREATE_AGENT: "Error al crear el agente",
  UPDATE_AGENT: "Error al actualizar el agente",
  DELETE_AGENT: "Error al eliminar el agente",
  TOGGLE_AGENT: "Error al cambiar el estado del agente",
  LOAD_AGENTS: "Error al cargar los agentes",
  LOAD_AGENT: "Error al cargar el agente",
  INVALID_JSON: "Los parámetros deben ser un JSON válido",
  MIN_FREQUENCY: (min: number) => `La frecuencia mínima es ${min} segundos`,
} as const;

export const UI_MESSAGES = {
  LOADING_AGENTS: "Cargando agentes...",
  NO_AGENTS: "No tienes agentes configurados aún",
  CREATE_FIRST_AGENT: "Crear tu primer agente",
  CREATE_NEW_AGENT: "Crear Nuevo Agente",
  DELETE_CONFIRMATION: "¿Estás seguro de que quieres eliminar este agente?",
  NEVER: "Nunca",
  NOT_SCHEDULED: "No programado",
  DELETING: "Eliminando...",
  DELETE: "Eliminar",
  CHANGING_STATE: "Cambiando estado...",
  CANCEL: "Cancelar",
  CREATE_AGENT: "Crear Agente",
} as const;







