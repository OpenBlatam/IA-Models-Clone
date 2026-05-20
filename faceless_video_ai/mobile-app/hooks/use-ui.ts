import { useState, useEffect } from 'react';
import { useColorScheme as useRNColorScheme, Appearance } from 'react-native';
import { useWindowDimensions } from 'react-native';
import { useAppStore } from '@/store/app-store';
import * as Haptics from 'expo-haptics';

export function useTheme() {
  const systemColorScheme = useRNColorScheme();
  const { theme, setTheme } = useAppStore();

  const effectiveTheme = theme === 'auto' ? systemColorScheme : theme;

  return {
    theme: effectiveTheme || 'light',
    isDark: effectiveTheme === 'dark',
    setTheme,
    systemTheme: systemColorScheme,
  };
}

export function useKeyboard() {
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);

  useEffect(() => {
    const showSubscription = Keyboard.addListener('keyboardDidShow', (e) => {
      setIsKeyboardVisible(true);
      setKeyboardHeight(e.endCoordinates.height);
    });
    const hideSubscription = Keyboard.addListener('keyboardDidHide', () => {
      setIsKeyboardVisible(false);
      setKeyboardHeight(0);
    });

    return () => {
      showSubscription.remove();
      hideSubscription.remove();
    };
  }, []);

  return { isKeyboardVisible, keyboardHeight };
}

export function useScreenDimensions() {
  const { width, height } = useWindowDimensions();
  const isSmallScreen = width < 375;
  const isMediumScreen = width >= 375 && width < 768;
  const isLargeScreen = width >= 768;
  const isTablet = width >= 768;

  return {
    width,
    height,
    isSmallScreen,
    isMediumScreen,
    isLargeScreen,
    isTablet,
  };
}

export function useHapticFeedback() {
  return {
    light: () => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light),
    medium: () => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium),
    heavy: () => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy),
    success: () => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success),
    error: () => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error),
    warning: () => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning),
  };
}

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

export function useThrottle<T>(value: T, limit: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastRan = React.useRef(Date.now());

  useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (Date.now() - lastRan.current));

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
}

export function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);

  const toggle = React.useCallback(() => setValue((v) => !v), []);
  const setTrue = React.useCallback(() => setValue(true), []);
  const setFalse = React.useCallback(() => setValue(false), []);

  return { value, toggle, setTrue, setFalse, setValue };
}

export function usePrevious<T>(value: T): T | undefined {
  const ref = React.useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

export function useInterval(callback: () => void, delay: number | null) {
  const savedCallback = React.useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    function tick() {
      savedCallback.current?.();
    }

    if (delay !== null) {
      const id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

export function useTimeout(callback: () => void, delay: number | null) {
  const savedCallback = React.useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay !== null) {
      const id = setTimeout(() => {
        savedCallback.current?.();
      }, delay);
      return () => clearTimeout(id);
    }
  }, [delay]);
}

import React from 'react';
import { Keyboard } from 'react-native';

