/**
 * Hook useSharedState
 * ===================
 * 
 * Hook para estado compartido entre componentes usando eventos
 */

import { useState, useEffect, useCallback, useRef } from 'react'

type SharedStateListener<T> = (value: T) => void

class SharedStateManager<T> {
  private state: T
  private listeners: Set<SharedStateListener<T>> = new Set()

  constructor(initialValue: T) {
    this.state = initialValue
  }

  getState(): T {
    return this.state
  }

  setState(value: T | ((prev: T) => T)): void {
    const newState = typeof value === 'function' 
      ? (value as (prev: T) => T)(this.state)
      : value
    
    if (newState !== this.state) {
      this.state = newState
      this.listeners.forEach(listener => listener(this.state))
    }
  }

  subscribe(listener: SharedStateListener<T>): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
}

const sharedStateManagers = new Map<string, SharedStateManager<any>>()

/**
 * Hook para estado compartido entre componentes
 */
export function useSharedState<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void] {
  const managerRef = useRef<SharedStateManager<T> | null>(null)

  if (!managerRef.current) {
    if (!sharedStateManagers.has(key)) {
      sharedStateManagers.set(key, new SharedStateManager(initialValue))
    }
    managerRef.current = sharedStateManagers.get(key) as SharedStateManager<T>
  }

  const [state, setState] = useState<T>(() => managerRef.current!.getState())

  useEffect(() => {
    const unsubscribe = managerRef.current!.subscribe((newState) => {
      setState(newState)
    })

    return unsubscribe
  }, [])

  const updateState = useCallback((value: T | ((prev: T) => T)) => {
    managerRef.current!.setState(value)
  }, [])

  return [state, updateState]
}

/**
 * Limpia un estado compartido
 */
export function clearSharedState(key: string): void {
  sharedStateManagers.delete(key)
}

/**
 * Limpia todos los estados compartidos
 */
export function clearAllSharedStates(): void {
  sharedStateManagers.clear()
}






