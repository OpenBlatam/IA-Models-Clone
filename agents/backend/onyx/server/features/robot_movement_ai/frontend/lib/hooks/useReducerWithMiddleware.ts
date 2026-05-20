import { useReducer, useCallback } from 'react';

type Middleware<T> = (action: T, next: (action: T) => void) => void;

export function useReducerWithMiddleware<TState, TAction>(
  reducer: (state: TState, action: TAction) => TState,
  initialState: TState,
  middlewares: Middleware<TAction>[] = []
) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const dispatchWithMiddleware = useCallback(
    (action: TAction) => {
      let index = 0;

      const next = (action: TAction) => {
        if (index < middlewares.length) {
          const middleware = middlewares[index++];
          middleware(action, next);
        } else {
          dispatch(action);
        }
      };

      next(action);
    },
    [middlewares]
  );

  return [state, dispatchWithMiddleware] as const;
}



