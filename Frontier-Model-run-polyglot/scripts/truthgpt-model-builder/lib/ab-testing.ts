/**
 * A/B Testing System
 * Sistema de pruebas A/B para configuraciones de modelos
 */

export interface ABTestVariant {
  id: string
  name: string
  config: Record<string, any>
  results?: ABTestResult[]
}

export interface ABTestResult {
  success: boolean
  duration: number
  timestamp: number
  [key: string]: any // For additional metrics like accuracy
}

export interface ABTestConfig {
  id: string
  name: string
  variants: ABTestVariant[]
  status: 'draft' | 'running' | 'completed' | 'paused'
  createdAt: number
}

export interface ABTestAnalysis {
  variants: Array<{
    id: string
    name: string
    successRate: number
    avgDuration: number
    totalRuns: number
    confidenceInterval?: [number, number]
  }>
  winner?: string
  confidence?: number
}

export class ABTesting {
  private tests: Map<string, ABTestConfig> = new Map()

  /**
   * Crear nuevo test A/B
   */
  createTest(config: {
    name: string
    variants: Array<{ id: string; name: string; config: Record<string, any> }>
  }): string {
    const id = `ab-test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const test: ABTestConfig = {
      id,
      name: config.name,
      variants: config.variants.map(v => ({
        ...v,
        results: [],
      })),
      status: 'draft',
      createdAt: Date.now(),
    }

    this.tests.set(id, test)
    return id
  }

  /**
   * Obtener test
   */
  getTest(testId: string): ABTestConfig | undefined {
    return this.tests.get(testId)
  }

  /**
   * Obtener todos los tests
   */
  getAllTests(): ABTestConfig[] {
    return Array.from(this.tests.values())
  }

  /**
   * Iniciar test
   */
  startTest(testId: string): void {
    const test = this.tests.get(testId)
    if (test) {
      test.status = 'running'
    }
  }

  /**
   * Registrar resultado
   */
  recordResult(
    testId: string,
    variantId: string,
    result: {
      success: boolean
      duration: number
      [key: string]: any
    }
  ): void {
    const test = this.tests.get(testId)
    if (!test) return

    const variant = test.variants.find(v => v.id === variantId)
    if (!variant) return

    if (!variant.results) {
      variant.results = []
    }

    variant.results.push({
      ...result,
      timestamp: Date.now(),
    })
  }

  /**
   * Analizar test
   */
  analyzeTest(testId: string): ABTestAnalysis {
    const test = this.tests.get(testId)
    if (!test) {
      return {
        variants: [],
      }
    }

    const variantStats = test.variants.map(variant => {
      const results = variant.results || []
      const successful = results.filter(r => r.success).length
      const totalRuns = results.length
      const successRate = totalRuns > 0 ? successful / totalRuns : 0
      const avgDuration =
        totalRuns > 0
          ? results.reduce((sum, r) => sum + (r.duration || 0), 0) / totalRuns
          : 0

      // Calculate confidence interval (simplified)
      const confidenceInterval: [number, number] = [
        Math.max(0, successRate - 0.1),
        Math.min(1, successRate + 0.1),
      ]

      return {
        id: variant.id,
        name: variant.name,
        successRate,
        avgDuration,
        totalRuns,
        confidenceInterval,
      }
    })

    // Determine winner
    let winner: string | undefined
    let confidence: number | undefined

    if (variantStats.length >= 2) {
      const sorted = [...variantStats].sort((a, b) => b.successRate - a.successRate)
      if (sorted[0].totalRuns > 0 && sorted[1].totalRuns > 0) {
        const diff = sorted[0].successRate - sorted[1].successRate
        if (diff > 0.05) {
          // 5% threshold
          winner = sorted[0].id
          confidence = Math.min(diff * 100, 100)
        }
      }
    }

    return {
      variants: variantStats,
      winner,
      confidence,
    }
  }

  /**
   * Completar test
   */
  completeTest(testId: string): void {
    const test = this.tests.get(testId)
    if (test) {
      test.status = 'completed'
    }
  }

  /**
   * Eliminar test
   */
  removeTest(testId: string): void {
    this.tests.delete(testId)
  }

  /**
   * Limpiar todos los tests
   */
  clear(): void {
    this.tests.clear()
  }
}

// Singleton instance
let abTestingInstance: ABTesting | null = null

export function getABTesting(): ABTesting {
  if (!abTestingInstance) {
    abTestingInstance = new ABTesting()
  }
  return abTestingInstance
}

export default ABTesting
