'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Search, Command, ArrowRight } from 'lucide-react'
import { Modal, Input } from './'
import { useKeyboardShortcut } from '@/hooks'

interface CommandItem {
  id: string
  label: string
  description?: string
  icon?: React.ReactNode
  action: () => void
  keywords?: string[]
  category?: string
}

interface CommandPaletteProps {
  commands: CommandItem[]
  isOpen?: boolean
  onClose?: () => void
  placeholder?: string
}

const CommandPalette: React.FC<CommandPaletteProps> = ({
  commands,
  isOpen: controlledIsOpen,
  onClose: controlledOnClose,
  placeholder = 'Type a command or search...',
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const isControlled = controlledIsOpen !== undefined

  const open = () => {
    if (!isControlled) {
      setIsOpen(true)
    }
  }

  const close = () => {
    setSearchQuery('')
    setSelectedIndex(0)
    if (isControlled) {
      controlledOnClose?.()
    } else {
      setIsOpen(false)
    }
  }

  useKeyboardShortcut({
    key: 'k',
    ctrl: true,
    callback: () => open(),
  })

  useEffect(() => {
    if (isOpen || controlledIsOpen) {
      inputRef.current?.focus()
    }
  }, [isOpen, controlledIsOpen])

  const filteredCommands = React.useMemo(() => {
    if (!searchQuery.trim()) return commands

    const query = searchQuery.toLowerCase()
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(query) ||
        cmd.description?.toLowerCase().includes(query) ||
        cmd.keywords?.some((kw) => kw.toLowerCase().includes(query)) ||
        cmd.category?.toLowerCase().includes(query)
    )
  }, [commands, searchQuery])

  const handleSelect = (command: CommandItem) => {
    command.action()
    close()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex((prev) =>
        prev < filteredCommands.length - 1 ? prev + 1 : prev
      )
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (filteredCommands[selectedIndex]) {
        handleSelect(filteredCommands[selectedIndex])
      }
    } else if (e.key === 'Escape') {
      close()
    }
  }

  useEffect(() => {
    if (listRef.current) {
      const selectedElement = listRef.current.children[selectedIndex] as HTMLElement
      selectedElement?.scrollIntoView({ block: 'nearest' })
    }
  }, [selectedIndex])

  const displayIsOpen = isControlled ? controlledIsOpen : isOpen

  return (
    <>
      <Modal
        isOpen={displayIsOpen}
        onClose={close}
        title=""
        size="lg"
        showCloseButton={false}
      >
        <div className="space-y-4">
          <div className="relative">
            <Input
              ref={inputRef}
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                setSelectedIndex(0)
              }}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              leftIcon={<Search className="w-4 h-4" />}
              className="pr-12"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 text-xs text-gray-500">
              <kbd className="px-2 py-1 bg-gray-100 rounded border border-gray-300">
                <Command className="w-3 h-3 inline" />
              </kbd>
              <span>+</span>
              <kbd className="px-2 py-1 bg-gray-100 rounded border border-gray-300">
                K
              </kbd>
            </div>
          </div>

          <div
            ref={listRef}
            className="max-h-96 overflow-y-auto space-y-1"
            role="listbox"
          >
            {filteredCommands.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No commands found
              </div>
            ) : (
              filteredCommands.map((command, index) => (
                <button
                  key={command.id}
                  onClick={() => handleSelect(command)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                    index === selectedIndex
                      ? 'bg-primary-50 border-2 border-primary-500'
                      : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
                  }`}
                  role="option"
                  aria-selected={index === selectedIndex}
                >
                  {command.icon && (
                    <div className="text-gray-600">{command.icon}</div>
                  )}
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{command.label}</p>
                    {command.description && (
                      <p className="text-sm text-gray-500">
                        {command.description}
                      </p>
                    )}
                  </div>
                  <ArrowRight className="w-4 h-4 text-gray-400" />
                </button>
              ))
            )}
          </div>
        </div>
      </Modal>
    </>
  )
}

export default CommandPalette

