/**
 * Configuration constants for Continuous Agent feature
 */

export const REFRESH_INTERVALS = {
  AGENTS_LIST: 10000,
  SINGLE_AGENT: 5000,
} as const;

export const TASK_TYPES = {
  CONTENT_GENERATION: "content_generation",
  DATA_PROCESSING: "data_processing",
  API_MONITORING: "api_monitoring",
  AUTOMATED_RESEARCH: "automated_research",
  CUSTOM: "custom",
} as const;

export const TASK_TYPE_LABELS: Record<string, string> = {
  [TASK_TYPES.CONTENT_GENERATION]: "Generación de Contenido",
  [TASK_TYPES.DATA_PROCESSING]: "Procesamiento de Datos",
  [TASK_TYPES.API_MONITORING]: "Monitoreo de API",
  [TASK_TYPES.AUTOMATED_RESEARCH]: "Investigación Automatizada",
  [TASK_TYPES.CUSTOM]: "Personalizado",
};

export const TASK_TYPE_OPTIONS = Object.entries(TASK_TYPE_LABELS).map(
  ([value, label]) => ({
    value,
    label,
  })
);

export const FORM_DEFAULTS = {
  TASK_TYPE: TASK_TYPES.CONTENT_GENERATION,
  FREQUENCY: 3600,
  PARAMETERS: "{}",
  GOAL: "",
  MIN_FREQUENCY: 60,
} as const;

export const FREQUENCY_EXAMPLES = {
  HOUR: 3600,
  DAY: 86400,
} as const;







