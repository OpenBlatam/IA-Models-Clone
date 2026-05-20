/**
 * Hook para testing y validación de modelos
 * ==========================================
 */

import { useState, useCallback } from 'react'

export interface TestResult {
  passed: boolean
  message: string
  details?: any
  duration?: number
}

export interface ModelTest {
  name: string
  description: string
  test: () => Promise<boolean> | boolean
}

export interface UseModelTestingResult {
  runTest: (test: ModelTest) => Promise<TestResult>
  runAllTests: (tests: ModelTest[]) => Promise<TestResult[]>
  testModelSpec: (spec: any) => TestResult[]
  testModelDescription: (description: string) => TestResult[]
  getTestSummary: (results: TestResult[]) => {
    total: number
    passed: number
    failed: number
    passRate: number
  }
}

/**
 * Hook para testing de modelos
 */
export function useModelTesting(): UseModelTestingResult {
  const [testResults, setTestResults] = useState<TestResult[]>([])

  const runTest = useCallback(async (test: ModelTest): Promise<TestResult> => {
    const startTime = Date.now()
    
    try {
      const result = await Promise.resolve(test.test())
      const duration = Date.now() - startTime
      
      const testResult: TestResult = {
        passed: result === true,
        message: result ? `✓ ${test.name} passed` : `✗ ${test.name} failed`,
        details: test.description,
        duration
      }

      setTestResults(prev => [...prev, testResult])
      return testResult
    } catch (error) {
      const duration = Date.now() - startTime
      const testResult: TestResult = {
        passed: false,
        message: `✗ ${test.name} failed: ${error instanceof Error ? error.message : String(error)}`,
        details: test.description,
        duration
      }

      setTestResults(prev => [...prev, testResult])
      return testResult
    }
  }, [])

  const runAllTests = useCallback(async (tests: ModelTest[]): Promise<TestResult[]> => {
    const results: TestResult[] = []
    
    for (const test of tests) {
      const result = await runTest(test)
      results.push(result)
    }

    return results
  }, [runTest])

  const testModelSpec = useCallback((spec: any): TestResult[] => {
    const results: TestResult[] = []

    // Test: spec debe ser un objeto
    results.push({
      passed: spec !== null && typeof spec === 'object' && !Array.isArray(spec),
      message: 'Spec debe ser un objeto',
      details: { spec }
    })

    // Test: layers debe ser array si existe
    if (spec?.layers) {
      results.push({
        passed: Array.isArray(spec.layers),
        message: 'Layers debe ser un array',
        details: { layers: spec.layers }
      })

      // Test: layers no debe estar vacío
      results.push({
        passed: spec.layers.length > 0,
        message: 'Layers no debe estar vacío',
        details: { layerCount: spec.layers.length }
      })
    }

    // Test: optimizer debe ser string válido
    if (spec?.optimizer) {
      const validOptimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw']
      results.push({
        passed: typeof spec.optimizer === 'string' && 
                validOptimizers.includes(spec.optimizer.toLowerCase()),
        message: 'Optimizer debe ser válido',
        details: { optimizer: spec.optimizer, validOptimizers }
      })
    }

    // Test: loss debe ser string válido
    if (spec?.loss) {
      const validLosses = [
        'sparsecategoricalcrossentropy',
        'categoricalcrossentropy',
        'binarycrossentropy',
        'meansquarederror',
        'mse',
        'meanabsoluteerror',
        'mae'
      ]
      results.push({
        passed: typeof spec.loss === 'string' && 
                validLosses.includes(spec.loss.toLowerCase()),
        message: 'Loss debe ser válido',
        details: { loss: spec.loss, validLosses }
      })
    }

    // Test: batch_size debe ser positivo
    if (spec?.batch_size !== undefined) {
      results.push({
        passed: typeof spec.batch_size === 'number' && spec.batch_size > 0,
        message: 'batch_size debe ser un número positivo',
        details: { batch_size: spec.batch_size }
      })
    }

    return results
  }, [])

  const testModelDescription = useCallback((description: string): TestResult[] => {
    const results: TestResult[] = []

    // Test: description debe ser string
    results.push({
      passed: typeof description === 'string',
      message: 'Description debe ser una cadena de texto',
      details: { type: typeof description }
    })

    // Test: description no debe estar vacío
    const trimmed = description.trim()
    results.push({
      passed: trimmed.length > 0,
      message: 'Description no debe estar vacío',
      details: { length: trimmed.length }
    })

    // Test: longitud mínima
    results.push({
      passed: trimmed.length >= 10,
      message: 'Description debe tener al menos 10 caracteres',
      details: { length: trimmed.length, minLength: 10 }
    })

    // Test: longitud máxima
    results.push({
      passed: trimmed.length <= 5000,
      message: 'Description no debe exceder 5000 caracteres',
      details: { length: trimmed.length, maxLength: 5000 }
    })

    // Test: debe contener palabras clave
    const hasKeywords = ['model', 'network', 'neural', 'classify', 'predict', 'regression'].some(
      keyword => trimmed.toLowerCase().includes(keyword)
    )
    results.push({
      passed: hasKeywords,
      message: 'Description debería contener palabras clave relevantes',
      details: { hasKeywords }
    })

    return results
  }, [])

  const getTestSummary = useCallback((results: TestResult[]) => {
    const total = results.length
    const passed = results.filter(r => r.passed).length
    const failed = total - passed
    const passRate = total > 0 ? (passed / total) * 100 : 0

    return {
      total,
      passed,
      failed,
      passRate
    }
  }, [])

  return {
    runTest,
    runAllTests,
    testModelSpec,
    testModelDescription,
    getTestSummary
  }
}

