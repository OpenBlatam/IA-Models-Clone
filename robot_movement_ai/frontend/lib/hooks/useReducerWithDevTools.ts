import { useReducer, useEffect } from 'react';

export function useReducerWithDevTools<TState, TAction>(
  reducer: (state: TState, action: TAction) => TState,
  initialState: TState,
  name: string = 'Reducer'
) {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && typeof window !== 'undefined') {
      // Log state changes in development
      console.log(`[${name}] State:`, state);
    }
  }, [state, name]);

  return [state, dispatch] as const;
}



