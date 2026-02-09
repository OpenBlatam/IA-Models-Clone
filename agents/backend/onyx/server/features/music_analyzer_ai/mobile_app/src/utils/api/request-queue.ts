import { AxiosResponse } from 'axios';

interface QueuedRequest {
  key: string;
  promise: Promise<AxiosResponse>;
  timestamp: number;
}

class RequestQueue {
  private queue: Map<string, QueuedRequest> = new Map();
  private readonly TTL = 5000;

  get<T>(key: string): Promise<AxiosResponse<T>> | null {
    const request = this.queue.get(key);

    if (!request) {
      return null;
    }

    const age = Date.now() - request.timestamp;
    if (age > this.TTL) {
      this.queue.delete(key);
      return null;
    }

    return request.promise as Promise<AxiosResponse<T>>;
  }

  set<T>(key: string, promise: Promise<AxiosResponse<T>>): void {
    this.queue.set(key, {
      key,
      promise: promise as Promise<AxiosResponse>,
      timestamp: Date.now(),
    });

    promise
      .then(() => {
        setTimeout(() => this.queue.delete(key), this.TTL);
      })
      .catch(() => {
        this.queue.delete(key);
      });
  }

  clear(): void {
    this.queue.clear();
  }

  size(): number {
    return this.queue.size;
  }
}

export const requestQueue = new RequestQueue();


