export async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function timeout<T>(
  promise: Promise<T>,
  ms: number,
  errorMessage?: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(
        () => reject(new Error(errorMessage || `Operation timed out after ${ms}ms`)),
        ms
      )
    ),
  ]);
}

export async function race<T>(
  promises: Promise<T>[]
): Promise<{ value: T; index: number }> {
  return Promise.race(
    promises.map((promise, index) =>
      promise.then((value) => ({ value, index }))
    )
  );
}

export function createAsyncQueue<T>() {
  const queue: Array<() => Promise<T>> = [];
  let running = false;

  async function processQueue(): Promise<void> {
    if (running || queue.length === 0) {
      return;
    }

    running = true;

    while (queue.length > 0) {
      const task = queue.shift();
      if (task) {
        try {
          await task();
        } catch (error) {
          console.error('Queue task error:', error);
        }
      }
    }

    running = false;
  }

  return {
    add: (task: () => Promise<T>): void => {
      queue.push(task);
      processQueue();
    },
    clear: (): void => {
      queue.length = 0;
    },
    get length(): number {
      return queue.length;
    },
  };
}

