'use client';

import { useState, KeyboardEvent, ChangeEvent } from 'react';
import { FiX } from 'react-icons/fi';
import { cn } from '@/utils/classNames';
import { Input } from './Input';
import { Badge } from './Badge';

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  maxTags?: number;
  className?: string;
  tagClassName?: string;
}

export function TagInput({
  tags,
  onChange,
  placeholder = 'Agregar etiqueta...',
  maxTags,
  className,
  tagClassName,
}: TagInputProps) {
  const [inputValue, setInputValue] = useState('');

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      addTag(inputValue.trim());
    } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
      removeTag(tags.length - 1);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const addTag = (tag: string) => {
    if (tag && !tags.includes(tag) && (!maxTags || tags.length < maxTags)) {
      onChange([...tags, tag]);
      setInputValue('');
    }
  };

  const removeTag = (index: number) => {
    onChange(tags.filter((_, i) => i !== index));
  };

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex flex-wrap gap-2 p-2 border border-gray-300 dark:border-gray-600 rounded-lg min-h-[42px]">
        {tags.map((tag, index) => (
          <Badge
            key={index}
            variant="primary"
            className={cn('flex items-center gap-1', tagClassName)}
          >
            {tag}
            <button
              onClick={() => removeTag(index)}
              className="ml-1 hover:opacity-70"
              aria-label="Eliminar etiqueta"
            >
              <FiX size={14} />
            </button>
          </Badge>
        ))}
        {(!maxTags || tags.length < maxTags) && (
          <Input
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 min-w-[120px] border-0 focus:ring-0 p-0 h-auto"
          />
        )}
      </div>
    </div>
  );
}

