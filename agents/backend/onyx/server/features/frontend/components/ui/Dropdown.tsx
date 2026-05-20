'use client';

import { ReactNode, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useClickOutside } from '@/hooks';
import { cn } from '@/utils/classNames';
import { FiChevronDown } from 'react-icons/fi';

interface DropdownOption {
  label: ReactNode;
  value: string;
  icon?: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}

interface DropdownProps {
  options: DropdownOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  trigger?: ReactNode;
  className?: string;
  align?: 'left' | 'right';
}

export function Dropdown({
  options,
  value,
  onChange,
  placeholder = 'Seleccionar...',
  trigger,
  className,
  align = 'left',
}: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useClickOutside<HTMLDivElement>(() => setIsOpen(false));

  const selectedOption = options.find((opt) => opt.value === value);

  const handleSelect = (option: DropdownOption) => {
    if (option.disabled) return;
    onChange?.(option.value);
    option.onClick?.();
    setIsOpen(false);
  };

  return (
    <div ref={dropdownRef} className={cn('relative', className)}>
      {trigger ? (
        <div onClick={() => setIsOpen(!isOpen)}>{trigger}</div>
      ) : (
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={cn(
            'w-full flex items-center justify-between px-4 py-2',
            'border border-gray-300 dark:border-gray-600 rounded-lg',
            'bg-white dark:bg-gray-800',
            'text-gray-900 dark:text-white',
            'hover:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
        >
          <span className={selectedOption ? '' : 'text-gray-500 dark:text-gray-400'}>
            {selectedOption ? selectedOption.label : placeholder}
          </span>
          <FiChevronDown
            size={18}
            className={cn(
              'text-gray-400 transition-transform',
              isOpen && 'transform rotate-180'
            )}
          />
        </button>
      )}

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={cn(
              'absolute z-50 mt-1 w-full min-w-[200px]',
              'bg-white dark:bg-gray-800',
              'border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg',
              'max-h-60 overflow-auto',
              align === 'right' ? 'right-0' : 'left-0'
            )}
          >
            {options.map((option) => (
              <button
                key={option.value}
                onClick={() => handleSelect(option)}
                disabled={option.disabled}
                className={cn(
                  'w-full flex items-center gap-2 px-4 py-2 text-left text-sm',
                  'hover:bg-gray-100 dark:hover:bg-gray-700',
                  'transition-colors',
                  option.value === value && 'bg-primary-50 dark:bg-primary-900/20',
                  option.disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                {option.icon && <span className="flex-shrink-0">{option.icon}</span>}
                <span className="flex-1">{option.label}</span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

