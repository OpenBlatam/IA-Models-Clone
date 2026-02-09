import { useEffect, useRef, useCallback } from 'react';
import { useFocusEffect as useRNFocusEffect } from '@react-navigation/native';

export const useFocusEffect = (
  callback: () => void | (() => void),
  deps?: React.DependencyList
) => {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback, ...(deps || [])]);

  useRNFocusEffect(
    useCallback(() => {
      return callbackRef.current();
    }, [])
  );
};

