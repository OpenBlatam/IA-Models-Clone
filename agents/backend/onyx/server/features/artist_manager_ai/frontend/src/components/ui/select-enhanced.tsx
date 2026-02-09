'use client';

import { forwardRef } from 'react';
import Select, { StylesConfig, SingleValue, MultiValue } from 'react-select';
import { cn } from '@/lib/utils';

interface Option {
  value: string;
  label: string;
}

interface SelectEnhancedProps {
  label?: string;
  error?: string;
  options: Option[];
  value?: string | string[];
  onChange: (value: string | string[]) => void;
  isMulti?: boolean;
  isClearable?: boolean;
  isSearchable?: boolean;
  placeholder?: string;
  className?: string;
  required?: boolean;
  disabled?: boolean;
}

const customStyles: StylesConfig<Option, boolean> = {
  control: (base, state) => ({
    ...base,
    borderColor: state.isFocused ? '#3b82f6' : state.hasError ? '#ef4444' : '#d1d5db',
    boxShadow: state.isFocused ? '0 0 0 2px rgba(59, 130, 246, 0.5)' : 'none',
    '&:hover': {
      borderColor: state.isFocused ? '#3b82f6' : '#9ca3af',
    },
  }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isSelected
      ? '#3b82f6'
      : state.isFocused
        ? '#eff6ff'
        : 'transparent',
    color: state.isSelected ? '#ffffff' : '#111827',
    '&:active': {
      backgroundColor: '#3b82f6',
      color: '#ffffff',
    },
  }),
};

const SelectEnhanced = forwardRef<HTMLDivElement, SelectEnhancedProps>(
  (
    {
      label,
      error,
      options,
      value,
      onChange,
      isMulti = false,
      isClearable = false,
      isSearchable = true,
      placeholder = 'Seleccionar...',
      className,
      required,
      disabled,
    },
    ref
  ) => {
    const handleChange = (selected: SingleValue<Option> | MultiValue<Option>) => {
      if (!selected) {
        onChange(isMulti ? [] : '');
        return;
      }

      if (isMulti && Array.isArray(selected)) {
        onChange(selected.map((option) => option.value));
      } else if (!isMulti && !Array.isArray(selected)) {
        onChange(selected.value);
      }
    };

    const getValue = () => {
      if (!value) {
        return null;
      }

      if (isMulti && Array.isArray(value)) {
        return options.filter((option) => value.includes(option.value));
      }

      return options.find((option) => option.value === value) || null;
    };

    return (
      <div className={cn('w-full', className)} ref={ref}>
        {label && (
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}
        <Select
          options={options}
          value={getValue()}
          onChange={handleChange}
          isMulti={isMulti}
          isClearable={isClearable}
          isSearchable={isSearchable}
          placeholder={placeholder}
          isDisabled={disabled}
          styles={customStyles}
          className={cn(error && 'border-red-500')}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={error ? `${label}-error` : undefined}
        />
        {error && (
          <p id={`${label}-error`} className="mt-1 text-sm text-red-600" role="alert">
            {error}
          </p>
        )}
      </div>
    );
  }
);

SelectEnhanced.displayName = 'SelectEnhanced';

export { SelectEnhanced };

