export const asyncUtils = {
  sleep: (ms: number): Promise<void> => {
    return new Promise((resolve) => setTimeout(resolve, ms))
  },

  timeout: <T>(promise: Promise<T>, ms: number): Promise<T> => {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) =>
        setTimeout(() => reject(new Error(`Operation timed out after ${ms}ms`)), ms)
      ),
    ])
  },

  retry: async <T>(
    fn: () => Promise<T>,
    retries = 3,
    delay = 1000
  ): Promise<T> => {
    try {
      return await fn()
    } catch (error) {
      if (retries === 0) {
        throw error
      }
      await asyncUtils.sleep(delay)
      return asyncUtils.retry(fn, retries - 1, delay)
    }
  },

  allSettled: async <T>(
    promises: Promise<T>[]
  ): Promise<Array<{ status: 'fulfilled' | 'rejected'; value?: T; reason?: Error }>> => {
    return Promise.all(
      promises.map((promise) =>
        promise
          .then((value) => ({ status: 'fulfilled' as const, value }))
          .catch((reason) => ({ status: 'rejected' as const, reason }))
      )
    )
  },
}

