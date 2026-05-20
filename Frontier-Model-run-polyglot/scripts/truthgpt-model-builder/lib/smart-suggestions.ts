/**
 * Smart Suggestions
 * Sistema inteligente de sugerencias basado en contexto
 */

export interface Suggestion {
  text: string
  category: string
  confidence: number
  keywords: string[]
}

export class SmartSuggestions {
  private history: string[] = []
  private maxHistorySize = 50

  /**
   * Agregar descripción al historial
   */
  addToHistory(description: string): void {
    if (!description || typeof description !== 'string') return
    
    const trimmed = description.trim()
    if (trimmed.length < 10) return

    this.history.push(trimmed)
    if (this.history.length > this.maxHistorySize) {
      this.history.shift()
    }
  }

  /**
   * Generar sugerencias basadas en entrada parcial
   */
  generateSuggestions(partialInput: string, maxSuggestions: number = 5): Suggestion[] {
    if (!partialInput || typeof partialInput !== 'string') return []
    
    const trimmed = partialInput.trim().toLowerCase()
    if (trimmed.length < 2) return []

    const suggestions: Suggestion[] = []

    // Analizar palabras clave comunes
    const commonPatterns = [
      { keywords: ['clasificación', 'clasificar', 'categorizar'], category: 'classification', text: 'Clasificador de texto para categorizar contenido' },
      { keywords: ['análisis', 'analizar', 'sentimiento'], category: 'sentiment', text: 'Analizador de sentimientos para texto' },
      { keywords: ['traducción', 'traducir', 'idioma'], category: 'translation', text: 'Modelo de traducción automática' },
      { keywords: ['generación', 'generar', 'texto'], category: 'generation', text: 'Generador de texto con IA' },
      { keywords: ['resumen', 'resumir', 'summarize'], category: 'summarization', text: 'Modelo para resumir documentos largos' },
      { keywords: ['pregunta', 'preguntar', 'qa'], category: 'qa', text: 'Sistema de preguntas y respuestas' },
      { keywords: ['imagen', 'imágenes', 'visual'], category: 'vision', text: 'Modelo de visión por computadora' },
      { keywords: ['audio', 'voz', 'speech'], category: 'audio', text: 'Modelo de procesamiento de audio' },
    ]

    // Buscar patrones coincidentes
    for (const pattern of commonPatterns) {
      const matches = pattern.keywords.filter(keyword => 
        trimmed.includes(keyword)
      ).length
      
      if (matches > 0) {
        const confidence = Math.min(matches / pattern.keywords.length, 1.0)
        suggestions.push({
          text: pattern.text,
          category: pattern.category,
          confidence,
          keywords: pattern.keywords,
        })
      }
    }

    // Buscar en historial
    for (const historyItem of this.history) {
      const historyLower = historyItem.toLowerCase()
      if (historyLower.includes(trimmed) || trimmed.includes(historyLower.substring(0, trimmed.length))) {
        const confidence = Math.min(trimmed.length / historyItem.length, 0.8)
        suggestions.push({
          text: historyItem,
          category: 'history',
          confidence,
          keywords: [],
        })
      }
    }

    // Ordenar por confianza y limitar
    const sorted = suggestions
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, maxSuggestions)

    // Asegurar que todas las sugerencias tengan categoría
    return sorted.map(s => ({
      ...s,
      category: s.category || 'general',
    }))
  }

  /**
   * Obtener sugerencias contextuales basadas en el último modelo
   */
  getContextualSuggestions(lastModel?: { description: string; category?: string }): Suggestion[] {
    if (!lastModel) return []

    const suggestions: Suggestion[] = []
    const desc = lastModel.description.toLowerCase()

    // Sugerencias relacionadas
    if (desc.includes('clasificación')) {
      suggestions.push({
        text: 'Clasificador multi-etiqueta para categorías múltiples',
        category: 'classification',
        confidence: 0.9,
        keywords: ['clasificación', 'multi-etiqueta'],
      })
    }

    if (desc.includes('análisis')) {
      suggestions.push({
        text: 'Analizador de sentimientos con detección de emociones',
        category: 'sentiment',
        confidence: 0.9,
        keywords: ['sentimiento', 'emociones'],
      })
    }

    return suggestions
  }

  /**
   * Limpiar historial
   */
  clearHistory(): void {
    this.history = []
  }

  /**
   * Limpiar todo (alias para compatibilidad)
   */
  clear(): void {
    this.clearHistory()
  }

  /**
   * Obtener historial
   */
  getHistory(): string[] {
    return [...this.history]
  }

  /**
   * Extraer patrones del historial
   */
  extractPatterns(): Array<{ pattern: string; frequency: number }> {
    const patterns: Map<string, number> = new Map()
    
    for (const item of this.history) {
      const words = item.toLowerCase().split(/\s+/)
      for (const word of words) {
        if (word.length > 3) {
          patterns.set(word, (patterns.get(word) || 0) + 1)
        }
      }
    }

    return Array.from(patterns.entries())
      .map(([pattern, frequency]) => ({ pattern, frequency }))
      .sort((a, b) => b.frequency - a.frequency)
  }
}

// Singleton instance
let smartSuggestionsInstance: SmartSuggestions | null = null

export function getSmartSuggestions(): SmartSuggestions {
  if (!smartSuggestionsInstance) {
    smartSuggestionsInstance = new SmartSuggestions()
  }
  return smartSuggestionsInstance
}

export default SmartSuggestions


