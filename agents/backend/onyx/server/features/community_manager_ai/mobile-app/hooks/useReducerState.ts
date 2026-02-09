import { useReducer, useCallback } from 'react';

/**
 * Generic reducer hook for complex state management
 */

interface Action<T> {
  type: string;
  payload?: T;
}

type Reducer<T> = (state: T, action: Action<any>) => T;

export function useReducerState<T>(
  reducer: Reducer<T>,
  initialState: T
): [T, (type: string, payload?: any) => void] {
  const [state, dispatch] = useReducer(reducer, initialState);

  const dispatchAction = useCallback((type: string, payload?: any) => {
    dispatch({ type, payload });
  }, []);

  return [state, dispatchAction];
}

/**
 * Create a typed reducer
 */
export function createReducer<T>(
  handlers: Record<string, (state: T, payload?: any) => T>
) {
  return (state: T, action: Action<any>): T => {
    const handler = handlers[action.type];
    if (handler) {
      return handler(state, action.payload);
    }
    return state;
  };
}


