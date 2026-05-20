type BatchCallback = () => void;

class Batcher {
  private callbacks: Set<BatchCallback> = new Set();
  private scheduled = false;

  add(callback: BatchCallback): () => void {
    this.callbacks.add(callback);

    if (!this.scheduled) {
      this.scheduled = true;
      requestAnimationFrame(() => {
        this.flush();
      });
    }

    return () => {
      this.callbacks.delete(callback);
    };
  }

  flush(): void {
    this.scheduled = false;
    const callbacks = Array.from(this.callbacks);
    this.callbacks.clear();
    callbacks.forEach((callback) => {
      try {
        callback();
      } catch (error) {
        console.error('Error in batch callback:', error);
      }
    });
  }
}

export const createBatcher = (): Batcher => {
  return new Batcher();
};

export const globalBatcher = createBatcher();

export const batch = (callback: BatchCallback): (() => void) => {
  return globalBatcher.add(callback);
};

