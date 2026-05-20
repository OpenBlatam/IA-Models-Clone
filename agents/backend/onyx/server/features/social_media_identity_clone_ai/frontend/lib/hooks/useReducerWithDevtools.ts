import { useReducer, useEffect, useRef } from 'react';

type Reducer<S, A> = (state: S, action: A) => S;

export const useReducerWithDevtools = <S, A>(
  reducer: Reducer<S, A>,
  initialState: S,
  name?: string
): [S, React.Dispatch<A>] => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const devtoolsRef = useRef<any>();

  useEffect(() => {
    if (typeof window === 'undefined' || !(window as any).__REDUX_DEVTOOLS_EXTENSION__) {
      return;
    }

    const devtools = (window as any).__REDUX_DEVTOOLS_EXTENSION__.connect({
      name: name || 'useReducer',
    });

    devtoolsRef.current = devtools;

    devtools.init(initialState);

    return () => {
      devtools.disconnect();
    };
  }, [initialState, name]);

  const dispatchWithDevtools = (action: A): void => {
    dispatch(action);
    if (devtoolsRef.current) {
      devtoolsRef.current.send(action, state);
    }
  };

  return [state, dispatchWithDevtools];
};



