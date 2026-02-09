export function batchProcess<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  batchSize = 5
): Promise<R[]> {
  const results: R[] = [];

  async function processBatch(startIndex: number): Promise<void> {
    const batch = items.slice(startIndex, startIndex + batchSize);
    const batchResults = await Promise.all(batch.map(processor));
    results.push(...batchResults);

    if (startIndex + batchSize < items.length) {
      // Yield to UI thread
      await new Promise((resolve) => setTimeout(resolve, 0));
      await processBatch(startIndex + batchSize);
    }
  }

  return processBatch(0).then(() => results);
}

export function createBatchProcessor<T, R>(
  processor: (item: T) => Promise<R>,
  batchSize = 5
) {
  return (items: T[]) => batchProcess(items, processor, batchSize);
}

