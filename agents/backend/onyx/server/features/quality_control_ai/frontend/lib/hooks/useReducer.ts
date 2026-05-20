import { useReducer as useReactReducer, useCallback } from 'react';

type ReducerAction<T> = 
  | { type: 'SET'; payload: T }
  | { type: 'RESET' }
  | { type: 'UPDATE'; payload: Partial<T> };

const createReducer = <T,>() => (
  state: T,
  action: ReducerAction<T>
): T => {
  switch (action.type) {
    case 'SET':
      return action.payload;
    case 'RESET':
      return state;
    case 'UPDATE':
      return { ...state, ...action.payload };
    default:
      return state;
  }
};

export const useReducer = <T extends Record<string, unknown>>(
  initialState: T
) => {
  const reducer = createReducer<T>();
  const [state, dispatch] = useReactReducer(reducer, initialState);

  const setState = useCallback((payload: T) => {
    dispatch({ type: 'SET', payload });
  }, []);

  const updateState = useCallback((payload: Partial<T>) => {
    dispatch({ type: 'UPDATE', payload });
  }, []);

  const resetState = useCallback(() => {
    dispatch({ type: 'RESET' });
  }, []);

  return {
    state,
    setState,
    updateState,
    resetState,
  };
};

