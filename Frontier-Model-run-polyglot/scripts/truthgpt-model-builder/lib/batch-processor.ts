/**
 * Batch Processor
 * Procesa múltiples modelos en paralelo con límite de concurrencia
 */

export interface BatchProcessorOptions {
  maxConcurrency?: number
  retryAttempts?: number
  retryDelay?: number
  onProgress?: (completed: number, total: number) => void
  onItemComplete?: (item: any, result: any) => void
  onItemError?: (item: any, error: Error) => void
}

export class BatchProcessor<T, R> {
  private maxConcurrency: number
  private retryAttempts: number
  private retryDelay: number
  private onProgress?: (completed: number, total: number) => void
  private onItemComplete?: (item: T, result: R) => void
  private onItemError?: (item: T, error: Error) => void

  constructor(options: BatchProcessorOptions = {}) {
    this.maxConcurrency = options.maxConcurrency || 3
    this.retryAttempts = options.retryAttempts || 3
    this.retryDelay = options.retryDelay || 1000
    this.onProgress = options.onProgress
    this.onItemComplete = options.onItemComplete
    this.onItemError = options.onItemError
  }

  /**
   * Procesar items en batch
   */
  async processBatch(
    items: T[],
    processor: (item: T) => Promise<R>
  ): Promise<{ results: R[]; errors: Array<{ item: T; error: Error }> }> {
    const results: R[] = []
    const errors: Array<{ item: T; error: Error }> = []
    let completed = 0

    // Procesar en chunks con límite de concurrencia
    for (let i = 0; i < items.length; i += this.maxConcurrency) {
      const chunk = items.slice(i, i + this.maxConcurrency)
      
      const chunkPromises = chunk.map(async (item) => {
        let lastError: Error | null = null
        
        // Retry logic
        for (let attempt = 0; attempt <= this.retryAttempts; attempt++) {
          try {
            const result = await processor(item)
            results.push(result)
            completed++
            
            if (this.onProgress) {
              this.onProgress(completed, items.length)
            }
            
            if (this.onItemComplete) {
              this.onItemComplete(item, result)
            }
            
            return
          } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error))
            
            if (attempt < this.retryAttempts) {
              // Exponential backoff
              await new Promise(resolve => 
                setTimeout(resolve, this.retryDelay * Math.pow(2, attempt))
              )
            }
          }
        }
        
        // Si todos los intentos fallaron
        if (lastError) {
          errors.push({ item, error: lastError })
          completed++
          
          if (this.onProgress) {
            this.onProgress(completed, items.length)
          }
          
          if (this.onItemError) {
            this.onItemError(item, lastError)
          }
        }
      })
      
      await Promise.all(chunkPromises)
    }

    return { results, errors }
  }

  /**
   * Procesar con límite de tiempo
   */
  async processWithTimeout(
    items: T[],
    processor: (item: T) => Promise<R>,
    timeout: number
  ): Promise<{ results: R[]; errors: Array<{ item: T; error: Error }> }> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('Batch processing timeout')), timeout)
    })

    const processPromise = this.processBatch(items, processor)

    try {
      return await Promise.race([processPromise, timeoutPromise])
    } catch (error) {
      // Si timeout, retornar resultados parciales
      return processPromise
    }
  }
}

export default BatchProcessor










