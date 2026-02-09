'use client';

import { memo } from 'react';
import { useHotkeys } from '@/lib/hooks';
import { cn } from '@/lib/utils';

interface KeyboardShortcut {
  keys: string[];
  description: string;
  action: () => void;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
}

interface KeyboardShortcutsProps {
  shortcuts: KeyboardShortcut[];
  className?: string;
}

const KeyboardShortcuts = memo(
  ({ shortcuts, className }: KeyboardShortcutsProps): JSX.Element => {
    useHotkeys(
      shortcuts.map((shortcut) => ({
        key: shortcut.keys[shortcut.keys.length - 1],
        ctrl: shortcut.ctrl,
        shift: shortcut.shift,
        alt: shortcut.alt,
        meta: shortcut.meta,
        callback: shortcut.action,
      }))
    );

    return null;
  }
);

KeyboardShortcuts.displayName = 'KeyboardShortcuts';

export default KeyboardShortcuts;

