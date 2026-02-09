'use client'

import clsx from 'clsx'

interface KeyboardShortcutProps {
  keys: string[]
  className?: string
}

const KeyboardShortcut = ({ keys, className }: KeyboardShortcutProps) => {
  return (
    <div className={clsx('flex items-center gap-1', className)}>
      {keys.map((key, index) => (
        <span key={index}>
          <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-300 rounded shadow-sm">
            {key}
          </kbd>
          {index < keys.length - 1 && <span className="mx-1 text-gray-400">+</span>}
        </span>
      ))}
    </div>
  )
}

export default KeyboardShortcut

