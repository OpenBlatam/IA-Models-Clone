/**
 * Benchmark utilities
 */

interface BenchmarkResult {
  name: string;
  duration: number;
  iterations: number;
  average: number;
}

// Benchmark a function
export function benchmark<T>(
  name: string,
  fn: () => T,
  iterations: number = 1000
): BenchmarkResult {
  const start = performance.now();

  for (let i = 0; i < iterations; i++) {
    fn();
  }

  const end = performance.now();
  const duration = end - start;
  const average = duration / iterations;

  const result: BenchmarkResult = {
    name,
    duration,
    iterations,
    average,
  };

  if (process.env.NODE_ENV === 'development') {
    console.log(`[Benchmark] ${name}:`, {
      total: `${duration.toFixed(2)}ms`,
      average: `${average.toFixed(4)}ms`,
      iterations,
    });
  }

  return result;
}

// Compare multiple functions
export function compareBenchmarks(
  functions: Array<{ name: string; fn: () => any }>,
  iterations: number = 1000
): BenchmarkResult[] {
  return functions.map(({ name, fn }) => benchmark(name, fn, iterations));
}

// Async benchmark
export async function benchmarkAsync<T>(
  name: string,
  fn: () => Promise<T>,
  iterations: number = 100
): Promise<BenchmarkResult> {
  const start = performance.now();

  for (let i = 0; i < iterations; i++) {
    await fn();
  }

  const end = performance.now();
  const duration = end - start;
  const average = duration / iterations;

  const result: BenchmarkResult = {
    name,
    duration,
    iterations,
    average,
  };

  if (process.env.NODE_ENV === 'development') {
    console.log(`[Benchmark] ${name}:`, {
      total: `${duration.toFixed(2)}ms`,
      average: `${average.toFixed(4)}ms`,
      iterations,
    });
  }

  return result;
}



