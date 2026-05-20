/**
 * Hook useReducerWithActions
 * ===========================
 * 
 * Hook para reducer con acciones tipadas
 */

import { useReducer, useCallback, useMemo } from 'react'

export type ActionCreator<T = any> = (...args: any[]) => { type: string; payload?: T }

export interface ActionCreators {
  [key: string]: ActionCreator
}

export interface UseReducerWithActionsReturn<TState, TActions extends ActionCreators> {
  state: TState
  actions: {
    [K in keyof TActions]: ReturnType<TActions[K]> extends { payload: infer P }
      ? (payload: P) => void
      : () => void
  }
  dispatch: React.Dispatch<any>
}

/**
 * Hook para reducer con acciones tipadas
 */
export function useReducerWithActions<TState, TActions extends ActionCreators>(
  reducer: (state: TState, action: any) => TState,
  initialState: TState,
  actionCreators: TActions
): UseReducerWithActionsReturn<TState, TActions> {
  const [state, dispatch] = useReducer(reducer, initialState)

  const actions = useMemo(() => {
    const boundActions: any = {}

    for (const [key, creator] of Object.entries(actionCreators)) {
      boundActions[key] = useCallback((...args: any[]) => {
        const action = creator(...args)
        dispatch(action)
      }, [creator])
    }

    return boundActions
  }, [actionCreators]) as UseReducerWithActionsReturn<TState, TActions>['actions']

  return {
    state,
    actions,
    dispatch
  }
}






