import { memo, useEffect } from 'react';
import { useKeyboardShortcuts } from '@/lib/hooks';
import { cn } from '@/lib/utils';

interface KeyboardShortcut {
  keys: string[];
  description: string;
}

interface KeyboardShortcutsProps {
  shortcuts: KeyboardShortcut[];
  className?: string;
}

const KeyboardShortcuts = memo(({ shortcuts, className = '' }: KeyboardShortcutsProps): JSX.Element => {
  const shortcutsMap = shortcuts.reduce((acc, shortcut) => {
    const key = shortcut.keys.join('+');
    acc[key] = () => {
      // Shortcuts are handled by useKeyboardShortcuts hook
    };
    return acc;
  }, {} as Record<string, () => void>);

  useKeyboardShortcuts(shortcutsMap);

  return (
    <div className={cn('space-y-2', className)}>
      <h3 className="text-sm font-semibold text-gray-700">Keyboard Shortcuts</h3>
      <div className="space-y-1">
        {shortcuts.map((shortcut, index) => (
          <div key={index} className="flex items-center justify-between text-sm">
            <span className="text-gray-600">{shortcut.description}</span>
            <div className="flex gap-1">
              {shortcut.keys.map((key, keyIndex) => (
                <kbd
                  key={keyIndex}
                  className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-xs font-mono"
                >
                  {key}
                </kbd>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});

KeyboardShortcuts.displayName = 'KeyboardShortcuts';

export default KeyboardShortcuts;



