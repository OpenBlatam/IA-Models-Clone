import React, { useEffect } from 'react';
import { Keyboard, Platform } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { hapticFeedback } from '../utils/haptics';

interface Shortcut {
  key: string;
  action: () => void;
  description: string;
}

export const useKeyboardShortcuts = (shortcuts: Shortcut[], enabled: boolean = true) => {
  const navigation = useNavigation();

  useEffect(() => {
    if (!enabled || Platform.OS !== 'ios') return;

    const handleKeyPress = (event: any) => {
      const pressedKey = event.nativeEvent.key;
      const shortcut = shortcuts.find((s) => s.key === pressedKey);
      
      if (shortcut) {
        event.preventDefault();
        hapticFeedback.selection();
        shortcut.action();
      }
    };

    const subscription = Keyboard.addListener('keyboardDidShow', () => {});
    
    return () => {
      subscription.remove();
    };
  }, [shortcuts, enabled]);
};

export const useAppShortcuts = () => {
  const navigation = useNavigation();

  const shortcuts: Shortcut[] = [
    {
      key: 'h',
      action: () => navigation.navigate('Home' as never),
      description: 'Ir a Inicio',
    },
    {
      key: 'p',
      action: () => navigation.navigate('Projects' as never),
      description: 'Ir a Proyectos',
    },
    {
      key: 'g',
      action: () => navigation.navigate('Generate' as never),
      description: 'Generar Proyecto',
    },
    {
      key: 's',
      action: () => navigation.navigate('Settings' as never),
      description: 'Configuración',
    },
  ];

  useKeyboardShortcuts(shortcuts, __DEV__);
};

