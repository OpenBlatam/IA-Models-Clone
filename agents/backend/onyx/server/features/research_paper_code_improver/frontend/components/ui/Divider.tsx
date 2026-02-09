import React from 'react'
import { clsx } from 'clsx'

interface DividerProps {
  orientation?: 'horizontal' | 'vertical'
  label?: string
  className?: string
}

const Divider: React.FC<DividerProps> = ({
  orientation = 'horizontal',
  label,
  className,
}) => {
  if (orientation === 'vertical') {
    return (
      <div
        className={clsx('w-px bg-gray-200 self-stretch', className)}
        role="separator"
        aria-orientation="vertical"
      />
    )
  }

  return (
    <div
      className={clsx('flex items-center gap-4 my-4', className)}
      role="separator"
      aria-orientation="horizontal"
    >
      <div className="flex-1 border-t border-gray-200" />
      {label && (
        <span className="text-sm text-gray-500 font-medium">{label}</span>
      )}
      <div className="flex-1 border-t border-gray-200" />
    </div>
  )
}

export default Divider




