'use client';

import { useState, KeyboardEvent } from 'react';
import { X } from 'lucide-react';
import { Badge } from './Badge';
import { Input } from './Input';
import { cn } from '@/lib/utils';

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  maxTags?: number;
  className?: string;
  disabled?: boolean;
}

export const TagInput = ({
  tags,
  onChange,
  placeholder = 'Agregar tag...',
  maxTags,
  className,
  disabled = false,
}: TagInputProps) => {
  const [inputValue, setInputValue] = useState('');

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      addTag(inputValue.trim());
    } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
      removeTag(tags.length - 1);
    }
  };

  const addTag = (tag: string) => {
    if (tag && !tags.includes(tag)) {
      if (maxTags && tags.length >= maxTags) {
        return;
      }
      onChange([...tags, tag]);
      setInputValue('');
    }
  };

  const removeTag = (index: number) => {
    onChange(tags.filter((_, i) => i !== index));
  };

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 p-2 min-h-[42px]">
        {tags.map((tag, index) => (
          <Badge
            key={index}
            variant="default"
            className="flex items-center gap-1 pr-1"
          >
            <span>{tag}</span>
            {!disabled && (
              <button
                type="button"
                onClick={() => removeTag(index)}
                className="ml-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 p-0.5 transition-colors"
                aria-label={`Eliminar tag ${tag}`}
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </Badge>
        ))}
        {(!maxTags || tags.length < maxTags) && (
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            className="flex-1 min-w-[120px] border-0 focus:ring-0 p-0 h-auto"
          />
        )}
      </div>
      {maxTags && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {tags.length} / {maxTags} tags
        </p>
      )}
    </div>
  );
};



