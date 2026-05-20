/**
 * Helpers para Testing
 * ====================
 * 
 * Utilidades para facilitar el testing
 */

// ============================================================================
// MOCKS Y FIXTURES
// ============================================================================

/**
 * Crea un mock de ModelState
 */
export function createMockModelState(overrides?: Partial<any>): any {
  return {
    id: 'model-123',
    name: 'Test Model',
    status: 'creating',
    progress: 50,
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-01'),
    ...overrides
  }
}

/**
 * Crea un mock de ModelSpec
 */
export function createMockModelSpec(overrides?: Partial<any>): any {
  return {
    type: 'classification',
    architecture: 'dense',
    layers: [
      {
        type: 'Dense',
        params: { units: 64 }
      }
    ],
    optimizer: {
      type: 'adam',
      learningRate: 0.001
    },
    loss: {
      type: 'categorical_crossentropy'
    },
    metrics: ['accuracy'],
    ...overrides
  }
}

/**
 * Crea un delay para testing
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Crea un mock de función
 */
export function createMockFn<T extends (...args: any[]) => any>(
  returnValue?: ReturnType<T>
): jest.Mock<T> {
  const mock = jest.fn<T, Parameters<T>>()
  if (returnValue !== undefined) {
    mock.mockReturnValue(returnValue as ReturnType<T>)
  }
  return mock
}

// ============================================================================
// ASSERTIONS
// ============================================================================

/**
 * Verifica que un objeto tenga las propiedades esperadas
 */
export function expectToHaveProperties<T>(
  obj: unknown,
  properties: Array<keyof T>
): asserts obj is T {
  expect(obj).toBeDefined()
  properties.forEach(prop => {
    expect(obj).toHaveProperty(String(prop))
  })
}

/**
 * Verifica que un array tenga elementos con cierta estructura
 */
export function expectArrayToHaveStructure<T>(
  array: unknown[],
  structure: Partial<T>
): void {
  expect(Array.isArray(array)).toBe(true)
  array.forEach(item => {
    Object.keys(structure).forEach(key => {
      expect(item).toHaveProperty(key)
    })
  })
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Limpia mocks después de cada test
 */
export function cleanupMocks(): void {
  jest.clearAllMocks()
}

/**
 * Espera a que una condición se cumpla
 */
export async function waitFor(
  condition: () => boolean,
  timeout: number = 5000,
  interval: number = 100
): Promise<void> {
  const start = Date.now()
  
  while (Date.now() - start < timeout) {
    if (condition()) {
      return
    }
    await delay(interval)
  }
  
  throw new Error(`Timeout waiting for condition after ${timeout}ms`)
}

/**
 * Crea un spy en un método de objeto
 */
export function spyOnMethod<T extends object, K extends keyof T>(
  obj: T,
  method: K
): jest.SpyInstance {
  return jest.spyOn(obj, method)
}







