'use client';

import { ReactNode } from 'react';

interface Hotkey {
  keys: string;
  callback: () => void;
  description?: string;
  enabled?: boolean;
}

interface HotkeysProps {
  hotkeys: Hotkey[];
  children?: ReactNode;
}

// Note: This component requires react-hotkeys-hook to be installed
// For now, it's a placeholder that can be extended
export function Hotkeys({ hotkeys, children }: HotkeysProps) {
  // Implementation would use useHotkeys hook for each hotkey
  // This is a simplified version that can be extended
  return <>{children}</>;
}

