import { useRef, useEffect } from 'react';

export const useUnmountRef = (): React.MutableRefObject<boolean> => {
  const unmountRef = useRef(false);

  useEffect(() => {
    return () => {
      unmountRef.current = true;
    };
  }, []);

  return unmountRef;
};

