'use client';

import React, { useState } from 'react';
import { ChevronRight, ChevronDown, File, Folder, FolderOpen } from 'lucide-react';
import { clsx } from 'clsx';

interface TreeNode {
  id: string;
  label: string;
  children?: TreeNode[];
  icon?: React.ReactNode;
  data?: any;
}

interface TreeViewProps {
  data: TreeNode[];
  onNodeSelect?: (node: TreeNode) => void;
  defaultExpanded?: string[];
  className?: string;
}

export const TreeView: React.FC<TreeViewProps> = ({
  data,
  onNodeSelect,
  defaultExpanded = [],
  className,
}) => {
  const [expanded, setExpanded] = useState<Set<string>>(
    new Set(defaultExpanded)
  );
  const [selected, setSelected] = useState<string | null>(null);

  const toggleExpanded = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleNodeClick = (node: TreeNode) => {
    setSelected(node.id);
    onNodeSelect?.(node);
    if (node.children && node.children.length > 0) {
      toggleExpanded(node.id);
    }
  };

  const renderNode = (node: TreeNode, level: number = 0) => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expanded.has(node.id);
    const isSelected = selected === node.id;

    return (
      <div key={node.id}>
        <div
          className={clsx(
            'flex items-center py-1 px-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-colors',
            isSelected && 'bg-primary-100 dark:bg-primary-900/20',
            className
          )}
          style={{ paddingLeft: `${level * 20 + 8}px` }}
          onClick={() => handleNodeClick(node)}
        >
          {hasChildren ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded(node.id);
              }}
              className="mr-1 p-0.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {isExpanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </button>
          ) : (
            <div className="w-5" />
          )}
          {node.icon || (
            <>
              {hasChildren ? (
                isExpanded ? (
                  <FolderOpen className="h-4 w-4 mr-2 text-gray-500" />
                ) : (
                  <Folder className="h-4 w-4 mr-2 text-gray-500" />
                )
              ) : (
                <File className="h-4 w-4 mr-2 text-gray-400" />
              )}
            </>
          )}
          <span
            className={clsx(
              'text-sm',
              isSelected
                ? 'text-primary-700 dark:text-primary-300 font-medium'
                : 'text-gray-700 dark:text-gray-300'
            )}
          >
            {node.label}
          </span>
        </div>
        {hasChildren && isExpanded && (
          <div>
            {node.children!.map((child) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full">
      {data.map((node) => renderNode(node))}
    </div>
  );
};


