/**
 * Componente FormField
 * ====================
 * 
 * Componente reutilizable para campos de formulario
 */

'use client'

import { InputHTMLAttributes, TextareaHTMLAttributes, SelectHTMLAttributes } from 'react'
import { UseFormResult } from '@/lib/hooks/useForm'

export interface FormFieldProps<T extends Record<string, unknown>> {
  form: UseFormResult<T>
  name: keyof T
  label?: string
  placeholder?: string
  type?: 'text' | 'email' | 'password' | 'number' | 'textarea' | 'select'
  options?: Array<{ value: string; label: string }>
  required?: boolean
  disabled?: boolean
  className?: string
  inputProps?: InputHTMLAttributes<HTMLInputElement> | TextareaHTMLAttributes<HTMLTextAreaElement> | SelectHTMLAttributes<HTMLSelectElement>
}

export function FormField<T extends Record<string, unknown>>({
  form,
  name,
  label,
  placeholder,
  type = 'text',
  options,
  required = false,
  disabled = false,
  className = '',
  inputProps
}: FormFieldProps<T>) {
  const fieldProps = form.getFieldProps(name)
  const error = form.errors[name]
  const touched = form.touched[name]

  const baseClasses = 'w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 transition-colors'
  const errorClasses = error && touched
    ? 'border-red-500 focus:ring-red-500'
    : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500'
  const disabledClasses = disabled ? 'bg-gray-100 dark:bg-gray-800 cursor-not-allowed' : 'bg-white dark:bg-gray-700'

  return (
    <div className={`space-y-1 ${className}`}>
      {label && (
        <label
          htmlFor={String(name)}
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}

      {type === 'textarea' ? (
        <textarea
          id={String(name)}
          value={fieldProps.value as string}
          onChange={(e) => fieldProps.onChange(e.target.value as T[keyof T])}
          onBlur={fieldProps.onBlur}
          placeholder={placeholder}
          disabled={disabled}
          required={required}
          className={`${baseClasses} ${errorClasses} ${disabledClasses} min-h-[100px] resize-y`}
          {...(inputProps as TextareaHTMLAttributes<HTMLTextAreaElement>)}
        />
      ) : type === 'select' ? (
        <select
          id={String(name)}
          value={fieldProps.value as string}
          onChange={(e) => fieldProps.onChange(e.target.value as T[keyof T])}
          onBlur={fieldProps.onBlur}
          disabled={disabled}
          required={required}
          className={`${baseClasses} ${errorClasses} ${disabledClasses}`}
          {...(inputProps as SelectHTMLAttributes<HTMLSelectElement>)}
        >
          {options?.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={String(name)}
          type={type}
          value={fieldProps.value as string | number}
          onChange={(e) => {
            const value = type === 'number' ? Number(e.target.value) : e.target.value
            fieldProps.onChange(value as T[keyof T])
          }}
          onBlur={fieldProps.onBlur}
          placeholder={placeholder}
          disabled={disabled}
          required={required}
          className={`${baseClasses} ${errorClasses} ${disabledClasses}`}
          {...(inputProps as InputHTMLAttributes<HTMLInputElement>)}
        />
      )}

      {error && touched && (
        <p className="text-sm text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}







