import { useState, useCallback, KeyboardEvent } from 'react';
import { cn } from '@/lib/utils';

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  className?: string;
  maxTags?: number;
}

const TagInput = ({
  tags,
  onChange,
  placeholder = 'Add tags...',
  className = '',
  maxTags,
}: TagInputProps): JSX.Element => {
  const [inputValue, setInputValue] = useState('');

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>): void => {
      if (e.key === 'Enter' && inputValue.trim()) {
        e.preventDefault();
        if (!maxTags || tags.length < maxTags) {
          if (!tags.includes(inputValue.trim())) {
            onChange([...tags, inputValue.trim()]);
          }
          setInputValue('');
        }
      } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
        onChange(tags.slice(0, -1));
      }
    },
    [inputValue, tags, onChange, maxTags]
  );

  const handleRemoveTag = useCallback(
    (tagToRemove: string): void => {
      onChange(tags.filter((tag) => tag !== tagToRemove));
    },
    [tags, onChange]
  );

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>): void => {
    setInputValue(e.target.value);
  }, []);

  return (
    <div
      className={cn(
        'flex flex-wrap gap-2 px-3 py-2 border border-gray-300 rounded-lg',
        'focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-transparent',
        className
      )}
    >
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center gap-1 px-2 py-1 bg-primary-100 text-primary-700 rounded text-sm"
        >
          {tag}
          <button
            type="button"
            onClick={() => handleRemoveTag(tag)}
            className="hover:text-primary-900"
            aria-label={`Remove tag ${tag}`}
            tabIndex={0}
          >
            ×
          </button>
        </span>
      ))}
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={tags.length === 0 ? placeholder : ''}
        className="flex-1 min-w-[120px] outline-none bg-transparent"
        disabled={maxTags ? tags.length >= maxTags : false}
        aria-label="Tag input"
      />
    </div>
  );
};

export default TagInput;



