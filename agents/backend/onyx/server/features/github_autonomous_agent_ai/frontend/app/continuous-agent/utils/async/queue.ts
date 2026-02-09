/**
 * Async queue utilities
 * 
 * Provides queue management for async operations with concurrency control
 */

/**
 * Queue options
 */
export interface QueueOptions {
  /** Maximum concurrent operations */
  readonly concurrency?: number;
  /** Whether to continue on error */
  readonly continueOnError?: boolean;
  /** Timeout for each operation in milliseconds */
  readonly timeout?: number;
}

/**
 * Queue item
 */
interface QueueItem<T> {
  readonly task: () => Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (error: unknown) => void;
}

/**
 * Async queue with concurrency control
 */
export class AsyncQueue<T = unknown> {
  private readonly queue: QueueItem<T>[] = [];
  private running = 0;
  private readonly concurrency: number;
  private readonly continueOnError: boolean;
  private readonly timeout?: number;

  constructor(options: QueueOptions = {}) {
    this.concurrency = options.concurrency ?? 1;
    this.continueOnError = options.continueOnError ?? false;
    this.timeout = options.timeout;
  }

  /**
   * Adds a task to the queue
   */
  async add(task: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.process();
    });
  }

  /**
   * Processes the queue
   */
  private async process(): Promise<void> {
    if (this.running >= this.concurrency || this.queue.length === 0) {
      return;
    }

    const item = this.queue.shift();
    if (!item) {
      return;
    }

    this.running++;

    try {
      let result: T;

      if (this.timeout) {
        result = await Promise.race([
          item.task(),
          new Promise<T>((_, reject) =>
            setTimeout(() => reject(new Error("Task timeout")), this.timeout)
          ),
        ]);
      } else {
        result = await item.task();
      }

      item.resolve(result);
    } catch (error) {
      item.reject(error);
      if (!this.continueOnError) {
        // Stop processing on error
        this.running--;
        return;
      }
    }

    this.running--;
    this.process();
  }

  /**
   * Gets queue size
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Gets number of running tasks
   */
  runningCount(): number {
    return this.running;
  }

  /**
   * Clears the queue
   */
  clear(): void {
    this.queue.forEach((item) => {
      item.reject(new Error("Queue cleared"));
    });
    this.queue.length = 0;
  }

  /**
   * Waits for all tasks to complete
   */
  async wait(): Promise<void> {
    while (this.running > 0 || this.queue.length > 0) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
}

/**
 * Creates an async queue
 */
export function createQueue<T = unknown>(options?: QueueOptions): AsyncQueue<T> {
  return new AsyncQueue<T>(options);
}

/**
 * Processes items in parallel with concurrency limit
 */
export async function parallelLimit<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  concurrency: number = 5
): Promise<R[]> {
  const queue = createQueue<R>({ concurrency });
  const promises = items.map((item) => queue.add(() => fn(item)));
  return Promise.all(promises);
}

/**
 * Processes items sequentially
 */
export async function sequential<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];
  for (const item of items) {
    results.push(await fn(item));
  }
  return results;
}




