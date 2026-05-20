import { useEffect, useRef } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback } from 'react';

export function useFocus(callback: () => void, deps: React.DependencyList = []) {
  useFocusEffect(
    useCallback(() => {
      callback();
    }, deps)
  );
}

export function useFocusRef() {
  const isFocusedRef = useRef(false);

  useFocusEffect(
    useCallback(() => {
      isFocusedRef.current = true;
      return () => {
        isFocusedRef.current = false;
      };
    }, [])
  );

  return isFocusedRef;
}


