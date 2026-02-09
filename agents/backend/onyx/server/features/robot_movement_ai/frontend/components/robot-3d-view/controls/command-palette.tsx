/**
 * Command Palette Component
 * @module robot-3d-view/controls/command-palette
 */

'use client';

import { memo, useState, useEffect, useRef } from 'react';
import { commandManager } from '../lib/command-system';
import { notify } from '../utils/notifications';

/**
 * Command Palette Component
 * 
 * Provides a CLI-like command interface.
 * 
 * @returns Command palette component
 */
export const CommandPalette = memo(() => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Open with Ctrl+K or Cmd+K
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }

      // Close with Escape
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        setInput('');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (input.trim()) {
      const autocomplete = commandManager.autocomplete(input);
      setSuggestions(autocomplete.slice(0, 5));
    } else {
      setSuggestions([]);
    }
  }, [input]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    try {
      const result = await commandManager.execute(input);
      if (result.success) {
        notify.success(result.message || 'Command executed successfully');
      } else {
        notify.error(result.error || 'Command failed');
      }
      setInput('');
      setIsOpen(false);
    } catch (error) {
      notify.error('Error executing command');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      const history = commandManager.getHistory();
      if (history.length > 0) {
        const newIndex = historyIndex < 0 ? history.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setInput(history[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      const history = commandManager.getHistory();
      if (historyIndex >= 0 && historyIndex < history.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setInput(history[newIndex]);
      } else {
        setHistoryIndex(-1);
        setInput('');
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="absolute inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={() => setIsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-4 max-w-2xl w-full mx-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              setHistoryIndex(-1);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Type a command... (try 'help')"
            className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            autoComplete="off"
          />
        </form>

        {suggestions.length > 0 && (
          <div className="mt-2 space-y-1">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => {
                  setInput(suggestion);
                  inputRef.current?.focus();
                }}
                className="w-full text-left px-4 py-2 bg-gray-700/50 hover:bg-gray-700 rounded text-gray-300 text-sm transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <div className="mt-2 text-xs text-gray-400">
          Press <kbd className="px-1 py-0.5 bg-gray-700 rounded">Esc</kbd> to close,{' '}
          <kbd className="px-1 py-0.5 bg-gray-700 rounded">↑↓</kbd> for history
        </div>
      </div>
    </div>
  );
});

CommandPalette.displayName = 'CommandPalette';



