import { useEffect, useRef } from 'react';

export const useMount = (callback: () => void) => {
  const hasMounted = useRef(false);

  useEffect(() => {
    if (!hasMounted.current) {
      hasMounted.current = true;
      callback();
    }
  }, [callback]);
};

