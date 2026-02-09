'use client';

import { useState } from 'react';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import * as Popover from '@radix-ui/react-popover';

interface ColorPickerProps {
  value?: string;
  onChange: (color: string) => void;
  colors?: string[];
  className?: string;
}

const defaultColors = [
  '#ef4444', '#f97316', '#f59e0b', '#eab308',
  '#84cc16', '#22c55e', '#10b981', '#14b8a6',
  '#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1',
  '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
  '#f43f5e', '#64748b', '#475569', '#334155',
];

export const ColorPicker = ({
  value = '#0ea5e9',
  onChange,
  colors = defaultColors,
  className,
}: ColorPickerProps) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Popover.Root open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={cn(
            'h-10 w-10 rounded-lg border-2 border-gray-300 dark:border-gray-700 transition-colors',
            'hover:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500',
            className
          )}
          style={{ backgroundColor: value }}
          aria-label="Seleccionar color"
        />
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          className={cn(
            'z-50 w-64 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-3 shadow-lg',
            'animate-in fade-in-0 zoom-in-95'
          )}
          sideOffset={5}
          align="start"
        >
          <div className="grid grid-cols-5 gap-2">
            {colors.map((color) => (
              <button
                key={color}
                type="button"
                onClick={() => {
                  onChange(color);
                  setIsOpen(false);
                }}
                className={cn(
                  'h-8 w-8 rounded-md transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary-500',
                  value === color && 'ring-2 ring-primary-500'
                )}
                style={{ backgroundColor: color }}
                aria-label={`Color ${color}`}
              >
                {value === color && (
                  <Check className="h-4 w-4 text-white mx-auto" />
                )}
              </button>
            ))}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
};



