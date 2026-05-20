/**
 * Contextual Help
 * Sistema de ayuda contextual
 */

export interface HelpTopic {
  id: string
  title: string
  content: string
  category: string
  tags: string[]
  relatedTopics?: string[]
  examples?: string[]
  videoUrl?: string
  documentationUrl?: string
}

export interface HelpContext {
  component?: string
  action?: string
  feature?: string
  error?: string
}

export class ContextualHelp {
  private topics: Map<string, HelpTopic> = new Map()
  private history: Array<{ topicId: string; timestamp: number }> = []

  /**
   * Agregar tópico de ayuda
   */
  addTopic(topic: HelpTopic): void {
    this.topics.set(topic.id, topic)
  }

  /**
   * Obtener tópico
   */
  getTopic(id: string): HelpTopic | undefined {
    return this.topics.get(id)
  }

  /**
   * Buscar tópicos por contexto
   */
  searchByContext(context: HelpContext): HelpTopic[] {
    const results: HelpTopic[] = []

    this.topics.forEach(topic => {
      let score = 0

      if (context.component && topic.tags.includes(context.component)) {
        score += 3
      }
      if (context.action && topic.tags.includes(context.action)) {
        score += 2
      }
      if (context.feature && topic.tags.includes(context.feature)) {
        score += 2
      }
      if (context.error && topic.tags.includes('error')) {
        score += 1
      }
      if (context.component && topic.category === context.component) {
        score += 1
      }

      if (score > 0) {
        results.push({ ...topic, ...{ relevanceScore: score } } as any)
      }
    })

    return results.sort((a, b) => {
      const scoreA = (a as any).relevanceScore || 0
      const scoreB = (b as any).relevanceScore || 0
      return scoreB - scoreA
    })
  }

  /**
   * Buscar tópicos
   */
  searchTopics(query: string): HelpTopic[] {
    const queryLower = query.toLowerCase()
    const results: HelpTopic[] = []

    this.topics.forEach(topic => {
      let score = 0

      if (topic.title.toLowerCase().includes(queryLower)) {
        score += 3
      }
      if (topic.content.toLowerCase().includes(queryLower)) {
        score += 2
      }
      if (topic.tags.some(tag => tag.toLowerCase().includes(queryLower))) {
        score += 1
      }
      if (topic.category.toLowerCase().includes(queryLower)) {
        score += 1
      }

      if (score > 0) {
        results.push({ ...topic, ...{ relevanceScore: score } } as any)
      }
    })

    return results.sort((a, b) => {
      const scoreA = (a as any).relevanceScore || 0
      const scoreB = (b as any).relevanceScore || 0
      return scoreB - scoreA
    })
  }

  /**
   * Obtener tópicos relacionados
   */
  getRelatedTopics(topicId: string): HelpTopic[] {
    const topic = this.topics.get(topicId)
    if (!topic || !topic.relatedTopics) return []

    return topic.relatedTopics
      .map(id => this.topics.get(id))
      .filter((t): t is HelpTopic => t !== undefined)
  }

  /**
   * Registrar visualización
   */
  recordView(topicId: string): void {
    this.history.push({
      topicId,
      timestamp: Date.now(),
    })

    // Limitar historial a 100
    if (this.history.length > 100) {
      this.history.shift()
    }
  }

  /**
   * Obtener tópicos más vistos
   */
  getMostViewedTopics(limit: number = 10): HelpTopic[] {
    const viewCounts = new Map<string, number>()

    this.history.forEach(entry => {
      viewCounts.set(entry.topicId, (viewCounts.get(entry.topicId) || 0) + 1)
    })

    return Array.from(viewCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([topicId]) => this.topics.get(topicId))
      .filter((t): t is HelpTopic => t !== undefined)
  }

  /**
   * Obtener todos los tópicos
   */
  getAllTopics(): HelpTopic[] {
    return Array.from(this.topics.values())
  }

  /**
   * Obtener tópicos por categoría
   */
  getTopicsByCategory(category: string): HelpTopic[] {
    return Array.from(this.topics.values()).filter(
      t => t.category === category
    )
  }

  /**
   * Obtener categorías
   */
  getCategories(): string[] {
    const categories = new Set<string>()
    this.topics.forEach(topic => {
      categories.add(topic.category)
    })
    return Array.from(categories).sort()
  }

  /**
   * Inicializar tópicos predefinidos
   */
  initializeDefaultTopics(): void {
    this.addTopic({
      id: 'getting-started',
      title: 'Comenzar',
      content: 'Guía para empezar a usar el Constructor Proactivo de TruthGPT.',
      category: 'general',
      tags: ['start', 'beginner', 'guide'],
      examples: [
        'Agregar modelo a la cola',
        'Iniciar construcción',
        'Ver resultados',
      ],
    })

    this.addTopic({
      id: 'proactive-builder',
      title: 'Constructor Proactivo',
      content: 'Construye modelos continuamente adaptados a optimization_core.',
      category: 'features',
      tags: ['proactive', 'builder', 'continuous'],
      relatedTopics: ['getting-started'],
    })

    this.addTopic({
      id: 'batch-mode',
      title: 'Modo Batch',
      content: 'Construye múltiples modelos en paralelo con configuración de concurrencia.',
      category: 'features',
      tags: ['batch', 'parallel', 'concurrency'],
    })

    this.addTopic({
      id: 'templates',
      title: 'Plantillas',
      content: 'Usa plantillas predefinidas o crea tus propias plantillas personalizadas.',
      category: 'features',
      tags: ['templates', 'custom', 'predefined'],
    })

    this.addTopic({
      id: 'statistics',
      title: 'Estadísticas',
      content: 'Visualiza estadísticas avanzadas y métricas de tus modelos.',
      category: 'features',
      tags: ['statistics', 'metrics', 'analytics'],
    })

    this.addTopic({
      id: 'keyboard-shortcuts',
      title: 'Atajos de Teclado',
      content: 'Atajos de teclado para navegar y usar el sistema rápidamente.',
      category: 'general',
      tags: ['shortcuts', 'keyboard', 'navigation'],
      examples: [
        'Ctrl+Enter: Iniciar construcción',
        'Ctrl+P: Pausar construcción',
        'Ctrl+K: Comandos rápidos',
      ],
    })
  }

  /**
   * Limpiar tópicos
   */
  clear(): void {
    this.topics.clear()
    this.history = []
  }
}

// Singleton instance
let contextualHelpInstance: ContextualHelp | null = null

export function getContextualHelp(): ContextualHelp {
  if (!contextualHelpInstance) {
    contextualHelpInstance = new ContextualHelp()
    contextualHelpInstance.initializeDefaultTopics()
  }
  return contextualHelpInstance
}

export default ContextualHelp










