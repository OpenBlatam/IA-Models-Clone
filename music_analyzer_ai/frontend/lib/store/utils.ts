/**
 * Store utility functions.
 * Helper functions for working with Zustand stores.
 */

import { type StoreApi, type UseBoundStore } from 'zustand';

/**
 * Subscribes to store changes and calls callback.
 * @param store - Zustand store
 * @param selector - Selector function
 * @param callback - Callback function
 * @returns Unsubscribe function
 */
export function subscribeToStore<T, U>(
  store: UseBoundStore<StoreApi<T>>,
  selector: (state: T) => U,
  callback: (selected: U) => void
): () => void {
  return store.subscribe(selector, callback);
}

/**
 * Gets current store state without subscribing.
 * @param store - Zustand store
 * @returns Current state
 */
export function getStoreState<T>(store: UseBoundStore<StoreApi<T>>): T {
  return store.getState();
}

/**
 * Resets store to initial state.
 * @param store - Zustand store
 * @param initialState - Initial state object
 */
export function resetStore<T>(
  store: UseBoundStore<StoreApi<T>>,
  initialState: Partial<T>
): void {
  const currentState = store.getState();
  store.setState({ ...currentState, ...initialState } as T);
}

