/**
 * Queue utility functions.
 * Provides helper functions for queue data structure.
 */

/**
 * Queue class implementation.
 */
export class Queue<T> {
  private items: T[] = [];

  /**
   * Adds item to queue.
   */
  enqueue(item: T): void {
    this.items.push(item);
  }

  /**
   * Removes and returns first item.
   */
  dequeue(): T | undefined {
    return this.items.shift();
  }

  /**
   * Returns first item without removing.
   */
  peek(): T | undefined {
    return this.items[0];
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
    return [...this.items];
  }
}

/**
 * Stack class implementation.
 */
export class Stack<T> {
  private items: T[] = [];

  /**
   * Adds item to stack.
   */
  push(item: T): void {
    this.items.push(item);
  }

  /**
   * Removes and returns last item.
   */
  pop(): T | undefined {
    return this.items.pop();
  }

  /**
   * Returns last item without removing.
   */
  peek(): T | undefined {
    return this.items[this.items.length - 1];
  }

  /**
   * Checks if stack is empty.
   */
  isEmpty(): boolean {
    return this.items.length === 0;
  }

  /**
   * Gets stack size.
   */
  size(): number {
    return this.items.length;
  }

  /**
   * Clears stack.
   */
  clear(): void {
    this.items = [];
  }

  /**
   * Converts stack to array.
   */
  toArray(): T[] {
    return [...this.items];
  }
}
