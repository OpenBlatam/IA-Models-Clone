/**
 * Decorator para agregar logging a servicios
 * Implementa el patrón Decorator para funcionalidades transversales
 */

export interface IServiceOperation<TArgs extends any[], TReturn> {
  (...args: TArgs): TReturn
}

export class LoggingDecorator {
  /**
   * Decora una función con logging
   */
  static withLogging<TArgs extends any[], TReturn>(
    operation: IServiceOperation<TArgs, TReturn>,
    operationName: string
  ): IServiceOperation<TArgs, TReturn> {
    return (...args: TArgs): TReturn => {
      console.log(`[${operationName}] Starting with args:`, args)
      const startTime = performance.now()
      
      try {
        const result = operation(...args)
        const duration = performance.now() - startTime
        console.log(`[${operationName}] Completed in ${duration.toFixed(2)}ms`)
        return result
      } catch (error) {
        const duration = performance.now() - startTime
        console.error(`[${operationName}] Failed after ${duration.toFixed(2)}ms:`, error)
        throw error
      }
    }
  }

  /**
   * Decora una función con logging y métricas
   */
  static withMetrics<TArgs extends any[], TReturn>(
    operation: IServiceOperation<TArgs, TReturn>,
    operationName: string
  ): IServiceOperation<TArgs, TReturn> {
    return this.withLogging(operation, operationName)
  }
}



