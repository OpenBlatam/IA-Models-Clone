/**
 * Utilidades de Testing
 * =====================
 * 
 * Funciones helper para testing
 */

/**
 * Mock de función con historial de llamadas
 */
export class MockFunction<T extends (...args: any[]) => any> {
  private calls: Array<{ args: Parameters<T>; result?: ReturnType<T> }> = []
  private implementation: T | null = null
  private returnValue: ReturnType<T> | null = null
  private throwError: Error | null = null

  constructor(implementation?: T) {
    if (implementation) {
      this.implementation = implementation
    }
  }

  /**
   * Ejecuta la función mock
   */
  (...args: Parameters<T>): ReturnType<T> {
    this.calls.push({ args })

    if (this.throwError) {
      throw this.throwError
    }

    if (this.implementation) {
      const result = this.implementation(...args)
      this.calls[this.calls.length - 1].result = result
      return result
    }

    return this.returnValue as ReturnType<T>
  }

  /**
   * Establece el valor de retorno
   */
  mockReturnValue(value: ReturnType<T>): this {
    this.returnValue = value
    this.throwError = null
    return this
  }

  /**
   * Establece una implementación
   */
  mockImplementation(fn: T): this {
    this.implementation = fn
    this.throwError = null
    return this
  }

  /**
   * Hace que la función lance un error
   */
  mockReject(error: Error): this {
    this.throwError = error
    this.returnValue = null
    this.implementation = null
    return this
  }

  /**
   * Obtiene todas las llamadas
   */
  getCalls(): Array<{ args: Parameters<T>; result?: ReturnType<T> }> {
    return [...this.calls]
  }

  /**
   * Obtiene la última llamada
   */
  getLastCall(): { args: Parameters<T>; result?: ReturnType<T> } | null {
    return this.calls[this.calls.length - 1] || null
  }

  /**
   * Obtiene el número de llamadas
   */
  getCallCount(): number {
    return this.calls.length
  }

  /**
   * Verifica si fue llamada
   */
  wasCalled(): boolean {
    return this.calls.length > 0
  }

  /**
   * Verifica si fue llamada con argumentos específicos
   */
  wasCalledWith(...args: Parameters<T>): boolean {
    return this.calls.some(call => {
      return call.args.length === args.length &&
        call.args.every((arg, index) => arg === args[index])
    })
  }

  /**
   * Resetea el mock
   */
  reset(): this {
    this.calls = []
    this.returnValue = null
    this.implementation = null
    this.throwError = null
    return this
  }
}

/**
 * Crea un mock de función
 */
export function createMockFunction<T extends (...args: any[]) => any>(
  implementation?: T
): MockFunction<T> {
  return new MockFunction(implementation)
}

/**
 * Mock de timer
 */
export class MockTimer {
  private timers: Map<number, { callback: () => void; delay: number; type: 'timeout' | 'interval' }> = new Map()
  private currentTime: number = 0
  private timerId: number = 0

  /**
   * Crea un timeout mock
   */
  setTimeout(callback: () => void, delay: number): number {
    const id = ++this.timerId
    this.timers.set(id, { callback, delay, type: 'timeout' })
    return id
  }

  /**
   * Crea un interval mock
   */
  setInterval(callback: () => void, delay: number): number {
    const id = ++this.timerId
    this.timers.set(id, { callback, delay, type: 'interval' })
    return id
  }

  /**
   * Limpia un timeout
   */
  clearTimeout(id: number): void {
    this.timers.delete(id)
  }

  /**
   * Limpia un interval
   */
  clearInterval(id: number): void {
    this.timers.delete(id)
  }

  /**
   * Avanza el tiempo
   */
  advanceTime(ms: number): void {
    this.currentTime += ms

    for (const [id, timer] of this.timers.entries()) {
      if (timer.delay <= this.currentTime) {
        timer.callback()
        if (timer.type === 'timeout') {
          this.timers.delete(id)
        }
      }
    }
  }

  /**
   * Obtiene el tiempo actual
   */
  getCurrentTime(): number {
    return this.currentTime
  }

  /**
   * Resetea el timer
   */
  reset(): void {
    this.timers.clear()
    this.currentTime = 0
    this.timerId = 0
  }
}

/**
 * Crea un mock de timer
 */
export function createMockTimer(): MockTimer {
  return new MockTimer()
}

/**
 * Espera un tiempo (útil para testing)
 */
export function wait(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Espera hasta que una condición sea verdadera
 */
export async function waitFor(
  condition: () => boolean,
  options: {
    timeout?: number
    interval?: number
  } = {}
): Promise<void> {
  const { timeout = 5000, interval = 100 } = options
  const start = Date.now()

  while (!condition()) {
    if (Date.now() - start > timeout) {
      throw new Error('Timeout waiting for condition')
    }
    await wait(interval)
  }
}

/**
 * Mock de fetch
 */
export class MockFetch {
  private responses: Map<string, Response | (() => Response)> = new Map()

  /**
   * Registra una respuesta para una URL
   */
  mockResponse(url: string, response: Response | (() => Response)): void {
    this.responses.set(url, response)
  }

  /**
   * Crea un fetch mock
   */
  createFetch(): typeof fetch {
    return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString()
      const response = this.responses.get(url)

      if (response) {
        return typeof response === 'function' ? response() : response
      }

      // Respuesta por defecto
      return new Response(JSON.stringify({}), {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' }
      })
    }
  }

  /**
   * Resetea el mock
   */
  reset(): void {
    this.responses.clear()
  }
}

/**
 * Crea un mock de fetch
 */
export function createMockFetch(): MockFetch {
  return new MockFetch()
}

/**
 * Genera datos de prueba
 */
export function generateTestData<T>(
  generator: (index: number) => T,
  count: number = 10
): T[] {
  return Array.from({ length: count }, (_, index) => generator(index))
}

/**
 * Crea un objeto de prueba con valores por defecto
 */
export function createTestObject<T extends Record<string, any>>(
  defaults: Partial<T>,
  overrides: Partial<T> = {}
): T {
  return { ...defaults, ...overrides } as T
}

/**
 * Verifica que un objeto tenga las propiedades esperadas
 */
export function expectObjectToHave<T extends Record<string, any>>(
  obj: T,
  properties: (keyof T)[]
): void {
  for (const prop of properties) {
    if (!(prop in obj)) {
      throw new Error(`Expected object to have property '${String(prop)}'`)
    }
  }
}

/**
 * Verifica que un array tenga elementos que cumplan una condición
 */
export function expectArrayToContain<T>(
  array: T[],
  predicate: (item: T) => boolean,
  count?: number
): void {
  const matches = array.filter(predicate)
  if (count !== undefined) {
    if (matches.length !== count) {
      throw new Error(`Expected array to contain ${count} items matching predicate, found ${matches.length}`)
    }
  } else {
    if (matches.length === 0) {
      throw new Error('Expected array to contain at least one item matching predicate')
    }
  }
}






