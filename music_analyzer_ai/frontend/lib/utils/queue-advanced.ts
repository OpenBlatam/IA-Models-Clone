/**
 * Advanced queue utility functions.
 * Provides helper functions for advanced queue operations.
 */

/**
 * Priority queue item.
 */
interface PriorityQueueItem<T> {
  item: T;
  priority: number;
}

/**
 * Priority queue implementation.
 */
export class PriorityQueue<T> {
  private items: PriorityQueueItem<T>[] = [];

  /**
   * Adds item with priority.
   */
  enqueue(item: T, priority: number): void {
    const queueItem: PriorityQueueItem<T> = { item, priority };
    let added = false;

    for (let i = 0; i < this.items.length; i++) {
      if (priority > this.items[i].priority) {
        this.items.splice(i, 0, queueItem);
        added = true;
        break;
      }
    }

    if (!added) {
      this.items.push(queueItem);
    }
  }

  /**
   * Removes and returns highest priority item.
   */
  dequeue(): T | undefined {
    return this.items.shift()?.item;
  }

  /**
   * Returns highest priority item without removing.
   */
  peek(): T | undefined {
    return this.items[0]?.item;
  }

  /**
   * Checks if queue is empty.
   */
  isEmpty(): boolean {
    return this.items.length === 0;
  }

  /**
   * Gets queue size.
   */
  size(): number {
    return this.items.length;
  }

  /**
   * Clears queue.
   */
  clear(): void {
    this.items = [];
  }

  /**
   * Converts queue to array.
   */
  toArray(): T[] {
    return this.items.map((item) => item.item);
  }
}

/**
 * Circular queue implementation.
 */
export class CircularQueue<T> {
  private items: (T | undefined)[];
  private front = 0;
  private rear = 0;
  private count = 0;

  constructor(private capacity: number) {
    this.items = new Array(capacity);
  }

  /**
   * Adds item to queue.
   */
  enqueue(item: T): boolean {
    if (this.isFull()) {
      return false;
    }

    this.items[this.rear] = item;
    this.rear = (this.rear + 1) % this.capacity;
    this.count++;
    return true;
  }

  /**
   * Removes and returns first item.
   */
  dequeue(): T | undefined {
    if (this.isEmpty()) {
      return undefined;
    }

    const item = this.items[this.front];
    this.items[this.front] = undefined;
    this.front = (this.front + 1) % this.capacity;
    this.count--;
    return item;
  }

  /**
   * Returns first item without removing.
   */
  peek(): T | undefined {
    return this.items[this.front];
  }

  /**
   * Checks if queue is empty.
   */
  isEmpty(): boolean {
    return this.count === 0;
  }

  /**
   * Checks if queue is full.
   */
  isFull(): boolean {
    return this.count === this.capacity;
  }

  /**
   * Gets queue size.
   */
  size(): number {
    return this.count;
  }

  /**
   * Clears queue.
   */
  clear(): void {
    this.items = new Array(this.capacity);
    this.front = 0;
    this.rear = 0;
    this.count = 0;
  }

  /**
   * Converts queue to array.
   */
  toArray(): T[] {
    const array: T[] = [];
    for (let i = 0; i < this.count; i++) {
      const index = (this.front + i) % this.capacity;
      const item = this.items[index];
      if (item !== undefined) {
        array.push(item);
      }
    }
    return array;
  }
}

