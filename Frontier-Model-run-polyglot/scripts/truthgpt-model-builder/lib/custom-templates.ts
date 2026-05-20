/**
 * Custom Templates
 * Sistema de plantillas personalizadas
 */

export interface CustomTemplate {
  id: string
  name: string
  description: string
  category: string
  tags: string[]
  spec: {
    architecture?: string
    parameters?: Record<string, any>
    training?: {
      learningRate?: number
      batchSize?: number
      epochs?: number
      optimizer?: string
    }
    data?: {
      dataset?: string
      preprocessing?: string[]
    }
  }
  example: string
  createdAt: number
  updatedAt: number
  usageCount: number
  isPublic: boolean
  author?: string
}

export class CustomTemplates {
  private templates: Map<string, CustomTemplate> = new Map()
  private storageKey = 'custom-model-templates'

  constructor() {
    this.loadFromStorage()
  }

  /**
   * Cargar desde almacenamiento
   */
  private loadFromStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const stored = window.localStorage.getItem(this.storageKey)
        if (stored) {
          const templates = JSON.parse(stored) as CustomTemplate[]
          templates.forEach(template => {
            this.templates.set(template.id, template)
          })
        }
      }
    } catch (error) {
      console.error('Error loading custom templates from storage:', error)
    }
  }

  /**
   * Guardar en almacenamiento
   */
  private saveToStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const templates = Array.from(this.templates.values())
        window.localStorage.setItem(this.storageKey, JSON.stringify(templates))
      }
    } catch (error) {
      console.error('Error saving custom templates to storage:', error)
    }
  }

  /**
   * Crear plantilla
   */
  createTemplate(config: {
    name: string
    description: string
    category: string
    spec: CustomTemplate['spec']
    example: string
    tags?: string[]
    isPublic?: boolean
    author?: string
  }): CustomTemplate {
    const id = `custom-template-${Date.now()}`
    const template: CustomTemplate = {
      id,
      name: config.name,
      description: config.description,
      category: config.category,
      tags: config.tags || [],
      spec: config.spec,
      example: config.example,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      usageCount: 0,
      isPublic: config.isPublic || false,
      author: config.author,
    }

    this.templates.set(id, template)
    this.saveToStorage()

    return template
  }

  /**
   * Obtener plantilla
   */
  getTemplate(id: string): CustomTemplate | undefined {
    return this.templates.get(id)
  }

  /**
   * Obtener todas las plantillas
   */
  getAllTemplates(): CustomTemplate[] {
    return Array.from(this.templates.values())
  }

  /**
   * Obtener plantillas por categoría
   */
  getTemplatesByCategory(category: string): CustomTemplate[] {
    return Array.from(this.templates.values()).filter(t => t.category === category)
  }

  /**
   * Buscar plantillas
   */
  searchTemplates(query: string): CustomTemplate[] {
    const queryLower = query.toLowerCase()
    return Array.from(this.templates.values()).filter(t =>
      t.name.toLowerCase().includes(queryLower) ||
      t.description.toLowerCase().includes(queryLower) ||
      t.tags.some(tag => tag.toLowerCase().includes(queryLower)) ||
      t.category.toLowerCase().includes(queryLower)
    )
  }

  /**
   * Actualizar plantilla
   */
  updateTemplate(id: string, updates: Partial<CustomTemplate>): CustomTemplate | null {
    const template = this.templates.get(id)
    if (!template) return null

    const updated: CustomTemplate = {
      ...template,
      ...updates,
      updatedAt: Date.now(),
    }

    this.templates.set(id, updated)
    this.saveToStorage()

    return updated
  }

  /**
   * Incrementar uso
   */
  incrementUsage(id: string): void {
    const template = this.templates.get(id)
    if (template) {
      template.usageCount++
      this.saveToStorage()
    }
  }

  /**
   * Eliminar plantilla
   */
  deleteTemplate(id: string): boolean {
    const deleted = this.templates.delete(id)
    if (deleted) {
      this.saveToStorage()
    }
    return deleted
  }

  /**
   * Exportar plantilla
   */
  exportTemplate(id: string): string {
    const template = this.templates.get(id)
    if (!template) return ''

    return JSON.stringify(template, null, 2)
  }

  /**
   * Importar plantilla
   */
  importTemplate(templateData: Partial<CustomTemplate>): CustomTemplate | null {
    try {
      // Validar estructura
      if (!templateData.name || !templateData.description || !templateData.spec) {
        throw new Error('Invalid template structure')
      }

      const id = templateData.id && !this.templates.has(templateData.id)
        ? templateData.id
        : `custom-template-${Date.now()}`

      const template: CustomTemplate = {
        id,
        name: templateData.name,
        description: templateData.description,
        category: templateData.category || 'general',
        tags: templateData.tags || [],
        spec: templateData.spec,
        example: templateData.example || '',
        createdAt: templateData.createdAt || Date.now(),
        updatedAt: Date.now(),
        usageCount: 0,
        isPublic: templateData.isPublic || false,
        author: templateData.author,
      }

      this.templates.set(template.id, template)
      this.saveToStorage()

      return template
    } catch (error) {
      console.error('Error importing template:', error)
      return null
    }
  }

  /**
   * Importar múltiples plantillas
   */
  importTemplates(templatesData: Array<Partial<CustomTemplate>>): CustomTemplate[] {
    const imported: CustomTemplate[] = []
    for (const templateData of templatesData) {
      const importedTemplate = this.importTemplate(templateData)
      if (importedTemplate) {
        imported.push(importedTemplate)
      }
    }
    return imported
  }

  /**
   * Exportar todas las plantillas
   */
  exportTemplates(): string {
    const templates = Array.from(this.templates.values())
    return JSON.stringify(templates, null, 2)
  }

  /**
   * Obtener plantillas más usadas
   */
  getMostUsedTemplates(limit: number = 10): CustomTemplate[] {
    return Array.from(this.templates.values())
      .sort((a, b) => b.usageCount - a.usageCount)
      .slice(0, limit)
  }

  /**
   * Obtener categorías
   */
  getCategories(): string[] {
    const categories = new Set<string>()
    this.templates.forEach(template => {
      if (template.category) {
        categories.add(template.category)
      }
    })
    return Array.from(categories).sort()
  }

  /**
   * Limpiar todas las plantillas
   */
  clear(): void {
    this.templates.clear()
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(this.storageKey)
    }
  }
}

// Singleton instance
let customTemplatesInstance: CustomTemplates | null = null

export function getCustomTemplates(): CustomTemplates {
  if (!customTemplatesInstance) {
    customTemplatesInstance = new CustomTemplates()
  }
  return customTemplatesInstance
}

export default CustomTemplates


