import { useState, useEffect } from 'react';
import { Keyboard, type KeyboardEvent } from 'react-native';

interface KeyboardState {
  isVisible: boolean;
  height: number;
}

/**
 * Hook to monitor keyboard visibility and height
 * Useful for adjusting UI when keyboard appears
 */
export function useKeyboard(): KeyboardState {
  const [keyboardState, setKeyboardState] = useState<KeyboardState>({
    isVisible: false,
    height: 0,
  });

  useEffect(() => {
    const showSubscription = Keyboard.addListener('keyboardDidShow', (e: KeyboardEvent) => {
      setKeyboardState({
        isVisible: true,
        height: e.endCoordinates.height,
      });
    });

    const hideSubscription = Keyboard.addListener('keyboardDidHide', () => {
      setKeyboardState({
        isVisible: false,
        height: 0,
      });
    });

    return () => {
      showSubscription.remove();
      hideSubscription.remove();
    };
  }, []);

  return keyboardState;
}

/**
 * Hook to dismiss keyboard
 */
export function useDismissKeyboard(): () => void {
  return () => {
    Keyboard.dismiss();
  };
}

