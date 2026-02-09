'use client'

import { useState, useCallback, useMemo, ReactNode, KeyboardEvent } from 'react'
import { Search, Command as CommandIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Input } from '@/components/ui'

interface CommandItem {
  id: string
  label: string
  icon?: ReactNode
  keywords?: string[]
  onSelect: () => void
  group?: string
}

interface CommandProps {
  items: CommandItem[]
  onClose?: () => void
  placeholder?: string
  className?: string
  showSearch?: boolean
  emptyMessage?: string
}

const Command = ({
  items,
  onClose,
  placeholder = 'Type a command or search...',
  className,
  showSearch = true,
  emptyMessage = 'No results found.',
}: CommandProps) => {
  const [search, setSearch] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)

  const filteredItems = useMemo(() => {
    if (!search) {
      return items
    }

    const searchLower = search.toLowerCase()
    return items.filter((item) => {
      const matchesLabel = item.label.toLowerCase().includes(searchLower)
      const matchesKeywords = item.keywords?.some((keyword) =>
        keyword.toLowerCase().includes(searchLower)
      )
      return matchesLabel || matchesKeywords
    })
  }, [items, search])

  const groupedItems = useMemo(() => {
    const groups: Record<string, CommandItem[]> = {}
    filteredItems.forEach((item) => {
      const group = item.group || 'Other'
      if (!groups[group]) {
        groups[group] = []
      }
      groups[group].push(item)
    })
    return groups
  }, [filteredItems])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex((prev) => Math.min(prev + 1, filteredItems.length - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex((prev) => Math.max(prev - 1, 0))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (filteredItems[selectedIndex]) {
          filteredItems[selectedIndex].onSelect()
          onClose?.()
        }
      } else if (e.key === 'Escape') {
        onClose?.()
      }
    },
    [filteredItems, selectedIndex, onClose]
  )

  const handleSelect = useCallback(
    (item: CommandItem) => {
      item.onSelect()
      onClose?.()
    },
    [onClose]
  )

  return (
    <div className={cn('bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden', className)}>
      {showSearch && (
        <div className="p-3 border-b border-gray-200">
          <Input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setSelectedIndex(0)
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            leftIcon={<Search className="w-4 h-4" />}
            className="w-full"
          />
        </div>
      )}
      <div className="max-h-96 overflow-y-auto">
        {Object.keys(groupedItems).length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <CommandIcon className="w-12 h-12 mx-auto mb-2 text-gray-400" />
            <p>{emptyMessage}</p>
          </div>
        ) : (
          Object.entries(groupedItems).map(([group, items]) => (
            <div key={group}>
              {group !== 'Other' && (
                <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase bg-gray-50">
                  {group}
                </div>
              )}
              {items.map((item, index) => {
                const globalIndex = filteredItems.indexOf(item)
                return (
                  <button
                    key={item.id}
                    onClick={() => handleSelect(item)}
                    className={cn(
                      'w-full flex items-center gap-3 px-3 py-2 text-sm text-left hover:bg-gray-50 transition-colors',
                      globalIndex === selectedIndex && 'bg-gray-100'
                    )}
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </button>
                )
              })}
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default Command

