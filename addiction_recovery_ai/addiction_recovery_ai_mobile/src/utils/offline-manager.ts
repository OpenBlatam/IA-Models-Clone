import NetInfo from '@react-native-community/netinfo';
import { cacheManager } from './cache-manager';

interface OfflineQueueItem {
  id: string;
  action: () => Promise<unknown>;
  timestamp: number;
}

class OfflineManager {
  private queue: OfflineQueueItem[] = [];
  private isOnline = true;
  private listeners: Array<(isOnline: boolean) => void> = [];

  constructor() {
    this.init();
  }

  private init(): void {
    NetInfo.addEventListener((state) => {
      const wasOnline = this.isOnline;
      this.isOnline = state.isConnected ?? false;

      if (wasOnline !== this.isOnline) {
        this.notifyListeners();
        
        if (this.isOnline) {
          this.processQueue();
        }
      }
    });

    NetInfo.fetch().then((state) => {
      this.isOnline = state.isConnected ?? false;
      this.notifyListeners();
    });
  }

  private notifyListeners(): void {
    this.listeners.forEach((listener) => listener(this.isOnline));
  }

  private async processQueue(): Promise<void> {
    while (this.queue.length > 0 && this.isOnline) {
      const item = this.queue.shift();
      if (item) {
        try {
          await item.action();
        } catch (error) {
          console.error('Failed to process offline queue item:', error);
          // Re-add to queue if it failed
          this.queue.push(item);
        }
      }
    }
  }

  addToQueue(action: () => Promise<unknown>): string {
    const id = `offline-${Date.now()}-${Math.random()}`;
    this.queue.push({
      id,
      action,
      timestamp: Date.now(),
    });
    return id;
  }

  subscribe(listener: (isOnline: boolean) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  getIsOnline(): boolean {
    return this.isOnline;
  }

  getQueueLength(): number {
    return this.queue.length;
  }
}

export const offlineManager = new OfflineManager();

