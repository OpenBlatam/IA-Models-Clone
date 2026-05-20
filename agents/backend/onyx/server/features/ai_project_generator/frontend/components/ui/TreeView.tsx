'use client'

import { useState, ReactNode, useCallback } from 'react'
import { ChevronRight, ChevronDown, Folder, FolderOpen, File } from 'lucide-react'
import { cn } from '@/lib/utils'

interface TreeNode {
  id: string
  label: string
  children?: TreeNode[]
  icon?: ReactNode
  data?: unknown
}

interface TreeViewProps {
  nodes: TreeNode[]
  onSelect?: (node: TreeNode) => void
  defaultExpanded?: string[]
  className?: string
  showIcons?: boolean
}

const TreeView = ({
  nodes,
  onSelect,
  defaultExpanded = [],
  className,
  showIcons = true,
}: TreeViewProps) => {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(defaultExpanded))
  const [selected, setSelected] = useState<string | null>(null)

  const toggleExpanded = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const handleSelect = useCallback(
    (node: TreeNode) => {
      setSelected(node.id)
      onSelect?.(node)
    },
    [onSelect]
  )

  const renderNode = (node: TreeNode, level = 0): ReactNode => {
    const hasChildren = node.children && node.children.length > 0
    const isExpanded = expanded.has(node.id)
    const isSelected = selected === node.id

    return (
      <div key={node.id}>
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-100 cursor-pointer',
            isSelected && 'bg-primary-100 text-primary-900',
            className
          )}
          style={{ paddingLeft: `${level * 1.5 + 0.5}rem` }}
          onClick={() => {
            if (hasChildren) {
              toggleExpanded(node.id)
            }
            handleSelect(node)
          }}
          role="treeitem"
          aria-expanded={hasChildren ? isExpanded : undefined}
          aria-selected={isSelected}
          tabIndex={0}
        >
          {hasChildren ? (
            <button
              onClick={(e) => {
                e.stopPropagation()
                toggleExpanded(node.id)
              }}
              className="p-0.5 hover:bg-gray-200 rounded"
              aria-label={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
          ) : (
            <div className="w-5" />
          )}
          {showIcons && (
            <div className="flex-shrink-0">
              {node.icon ||
                (hasChildren ? (
                  isExpanded ? (
                    <FolderOpen className="w-4 h-4 text-blue-600" />
                  ) : (
                    <Folder className="w-4 h-4 text-blue-600" />
                  )
                ) : (
                  <File className="w-4 h-4 text-gray-600" />
                ))}
            </div>
          )}
          <span className="flex-1">{node.label}</span>
        </div>
        {hasChildren && isExpanded && (
          <div role="group">
            {node.children!.map((child) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="border rounded-lg p-2" role="tree">
      {nodes.map((node) => renderNode(node))}
    </div>
  )
}

export default TreeView

