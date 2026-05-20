'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import clsx from 'clsx'
import { ChevronDown } from 'lucide-react'
import { useClickOutside } from '@/hooks/ui'
import Button from './Button'

interface DropdownOption {
  value: string
  label: string
  icon?: React.ReactNode
  disabled?: boolean
}

interface DropdownProps {
  options: DropdownOption[]
  value?: string
  onChange: (value: string) => void
  placeholder?: string
  label?: string
  className?: string
  disabled?: boolean
}

const Dropdown = ({ options, value, onChange, placeholder = 'Select...', label, className, disabled }: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useClickOutside<HTMLDivElement>(() => setIsOpen(false))

  const selectedOption = options.find((opt) => opt.value === value)

  const handleSelect = useCallback(
    (optionValue: string) => {
      if (!options.find((opt) => opt.value === optionValue)?.disabled) {
        onChange(optionValue)
        setIsOpen(false)
      }
    },
    [onChange, options]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsOpen(false)
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        setIsOpen(!isOpen)
      }
    },
    [isOpen]
  )

  return (
    <div className={clsx('relative', className)} ref={dropdownRef}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
      )}
      <Button
        variant="secondary"
        size="md"
        rightIcon={<ChevronDown className={clsx('w-4 h-4 transition-transform', isOpen && 'rotate-180')} />}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className="w-full justify-between"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        {selectedOption ? (
          <span className="flex items-center gap-2">
            {selectedOption.icon}
            {selectedOption.label}
          </span>
        ) : (
          <span className="text-gray-500">{placeholder}</span>
        )}
      </Button>

      {isOpen && (
        <div
          className="absolute z-50 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-auto"
          role="listbox"
        >
          {options.map((option) => (
            <button
              key={option.value}
              onClick={() => handleSelect(option.value)}
              disabled={option.disabled}
              className={clsx(
                'w-full px-4 py-2 text-left flex items-center gap-2 hover:bg-gray-50 transition-colors',
                value === option.value && 'bg-primary-50 text-primary-700',
                option.disabled && 'opacity-50 cursor-not-allowed'
              )}
              role="option"
              aria-selected={value === option.value}
              tabIndex={0}
            >
              {option.icon}
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default Dropdown

