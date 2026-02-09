import { memo, useState, useEffect, useRef } from 'react';
import { useKeyboardShortcuts } from '@/lib/hooks';
import Modal from './Modal';
import Input from './Input';
import { cn } from '@/lib/utils';
import { Search } from 'lucide-react';

interface Command {
  id: string;
  label: string;
  description?: string;
  shortcut?: string;
  action: () => void;
}

interface CommandPaletteProps {
  commands: Command[];
  triggerKey?: string;
  className?: string;
}

const CommandPalette = memo(({
  commands,
  triggerKey = 'k',
  className = '',
}: CommandPaletteProps): JSX.Element => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredCommands = commands.filter((command) =>
    command.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    command.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  useKeyboardShortcuts({
    [`Meta+${triggerKey}`]: () => setIsOpen(true),
    [`Control+${triggerKey}`]: () => setIsOpen(true),
  });

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      setSearchQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const handleSelect = (command: Command): void => {
    command.action();
    setIsOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredCommands.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filteredCommands[selectedIndex]) {
        handleSelect(filteredCommands[selectedIndex]);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <>
      <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} className={className}>
        <div className="space-y-4">
          <div className="flex items-center gap-2 border-b pb-2">
            <Search className="w-5 h-5 text-gray-400" />
            <Input
              ref={inputRef}
              type="text"
              placeholder="Type to search commands..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="border-0 focus:ring-0"
            />
          </div>

          <div className="max-h-64 overflow-y-auto">
            {filteredCommands.length === 0 ? (
              <div className="text-center text-gray-500 py-8">No commands found</div>
            ) : (
              <div className="space-y-1">
                {filteredCommands.map((command, index) => (
                  <button
                    key={command.id}
                    onClick={() => handleSelect(command)}
                    className={cn(
                      'w-full text-left px-4 py-2 rounded hover:bg-gray-100 transition-colors',
                      index === selectedIndex && 'bg-gray-100'
                    )}
                  >
                    <div className="font-medium">{command.label}</div>
                    {command.description && (
                      <div className="text-sm text-gray-500">{command.description}</div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </Modal>
    </>
  );
});

CommandPalette.displayName = 'CommandPalette';

export default CommandPalette;



