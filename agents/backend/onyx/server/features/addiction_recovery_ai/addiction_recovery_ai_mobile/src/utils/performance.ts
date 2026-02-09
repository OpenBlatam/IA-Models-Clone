import { InteractionManager } from 'react-native';

export function runAfterInteractions(callback: () => void): void {
  InteractionManager.runAfterInteractions(callback);
}

export function createInteractionHandle(): number {
  return InteractionManager.createInteractionHandle();
}

export function clearInteractionHandle(handle: number): void {
  InteractionManager.clearInteractionHandle(handle);
}

export function requestIdleCallback(callback: () => void, timeout = 5000): number {
  if (typeof requestIdleCallback !== 'undefined') {
    return window.requestIdleCallback(callback, { timeout });
  }
  
  return setTimeout(callback, timeout) as unknown as number;
}

export function cancelIdleCallback(handle: number): void {
  if (typeof cancelIdleCallback !== 'undefined') {
    window.cancelIdleCallback(handle);
  } else {
    clearTimeout(handle);
  }
}

