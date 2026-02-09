import { produce } from 'immer';
import type { StateCreator } from 'zustand';

export function withImmer<T extends object>(
  config: StateCreator<T>
): StateCreator<T> {
  return (set, get, api) =>
    config(
      (fn) => set(produce(fn) as any),
      get,
      api
    );
}

