/**
 * Management Module - Exports all management-related functionality
 */

export { 
  analyzeModelDescription, 
  analyzeModelDescriptionLegacy,
  generateArchitectureCode,
  type ModelSpec 
} from './model-analyzer'

export { ModelManager, type ModelInfo } from './model-manager'
export { webhookManager, type WebhookEvent, type WebhookConfig } from '../../webhooks'
export { saveVersion, getVersionHistory, generateNextVersion, cloneModelAsVersion, type ModelVersion, type VersionHistory } from '../../versioning'
export { ModelVersioning, getModelVersioning, type ModelVersion as ModelVersionType } from './model-versioning'
export { ModelExporter, getModelExporter, type ExportFormat, type ExportOptions } from './model-exporter'
export {
  MODEL_TEMPLATES,
  getTemplatesByCategory,
  getTemplatesByTag,
  searchTemplates,
  getTemplateById,
  getAllCategories,
  getAllTags,
  type ModelTemplate,
} from './model-templates'

export * from './model-optimizer'
export * from './model-validator'

