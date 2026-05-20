'use client';

import React, { ReactNode } from 'react';
import { useDefaultShortcuts } from '@/hooks/useKeyboardShortcuts';

interface KeyboardShortcutsProviderProps {
  children: ReactNode;
}

export const KeyboardShortcutsProvider: React.FC<KeyboardShortcutsProviderProps> = ({
  children,
}) => {
  useDefaultShortcuts();
  return <>{children}</>;
};

