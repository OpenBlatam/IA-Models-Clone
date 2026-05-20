/**
 * Hook useStateMachine
 * =====================
 * 
 * Hook para máquinas de estado finitas
 */

import { useState, useCallback, useMemo } from 'react'

export type StateTransition<TState extends string, TEvent extends string> = {
  from: TState
  to: TState
  event: TEvent
  guard?: () => boolean
  action?: () => void
}

export interface StateMachineConfig<TState extends string, TEvent extends string> {
  initialState: TState
  states: TState[]
  transitions: StateTransition<TState, TEvent>[]
  onStateChange?: (from: TState, to: TState, event: TEvent) => void
}

export interface UseStateMachineReturn<TState extends string, TEvent extends string> {
  state: TState
  canTransition: (event: TEvent) => boolean
  transition: (event: TEvent) => boolean
  reset: () => void
  isInState: (state: TState) => boolean
}

/**
 * Hook para máquinas de estado finitas
 */
export function useStateMachine<TState extends string, TEvent extends string>(
  config: StateMachineConfig<TState, TEvent>
): UseStateMachineReturn<TState, TEvent> {
  const { initialState, transitions, onStateChange } = config
  const [state, setState] = useState<TState>(initialState)

  const transitionsByEvent = useMemo(() => {
    const map = new Map<TEvent, StateTransition<TState, TEvent>[]>()
    
    for (const transition of transitions) {
      if (!map.has(transition.event)) {
        map.set(transition.event, [])
      }
      map.get(transition.event)!.push(transition)
    }
    
    return map
  }, [transitions])

  const canTransition = useCallback((event: TEvent): boolean => {
    const possibleTransitions = transitionsByEvent.get(event) || []
    
    return possibleTransitions.some(transition => {
      if (transition.from !== state) {
        return false
      }
      if (transition.guard && !transition.guard()) {
        return false
      }
      return true
    })
  }, [state, transitionsByEvent])

  const transition = useCallback((event: TEvent): boolean => {
    const possibleTransitions = transitionsByEvent.get(event) || []
    
    for (const transition of possibleTransitions) {
      if (transition.from !== state) {
        continue
      }
      
      if (transition.guard && !transition.guard()) {
        continue
      }

      const fromState = state
      const toState = transition.to

      if (transition.action) {
        transition.action()
      }

      setState(toState)
      onStateChange?.(fromState, toState, event)
      
      return true
    }

    return false
  }, [state, transitionsByEvent, onStateChange])

  const reset = useCallback(() => {
    setState(initialState)
  }, [initialState])

  const isInState = useCallback((checkState: TState): boolean => {
    return state === checkState
  }, [state])

  return {
    state,
    canTransition,
    transition,
    reset,
    isInState
  }
}






