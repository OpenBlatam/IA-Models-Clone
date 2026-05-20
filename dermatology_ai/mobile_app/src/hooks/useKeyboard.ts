import { useState, useEffect } from 'react';
import { Keyboard, KeyboardEvent } from 'react-native';

interface KeyboardState {
  isVisible: boolean;
  height: number;
}

/**
 * Hook to track keyboard visibility and height
 */
export const useKeyboard = (): KeyboardState => {
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
};

/**
 * Hook to dismiss keyboard
 */
export const useDismissKeyboard = () => {
  const dismiss = () => {
    Keyboard.dismiss();
  };

  return { dismiss };
};
