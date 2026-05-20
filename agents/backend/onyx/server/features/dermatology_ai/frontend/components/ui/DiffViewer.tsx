'use client';

import React from 'react';
import { clsx } from 'clsx';

interface DiffViewerProps {
  oldText: string;
  newText: string;
  className?: string;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({
  oldText,
  newText,
  className,
}) => {
  const computeDiff = (oldText: string, newText: string) => {
    const oldLines = oldText.split('\n');
    const newLines = newText.split('\n');
    const diff: Array<{ type: 'added' | 'removed' | 'unchanged'; line: string; oldLineNumber?: number; newLineNumber?: number }> = [];

    let oldIndex = 0;
    let newIndex = 0;

    while (oldIndex < oldLines.length || newIndex < newLines.length) {
      if (oldIndex >= oldLines.length) {
        diff.push({ type: 'added', line: newLines[newIndex], newLineNumber: newIndex + 1 });
        newIndex++;
      } else if (newIndex >= newLines.length) {
        diff.push({ type: 'removed', line: oldLines[oldIndex], oldLineNumber: oldIndex + 1 });
        oldIndex++;
      } else if (oldLines[oldIndex] === newLines[newIndex]) {
        diff.push({ type: 'unchanged', line: oldLines[oldIndex], oldLineNumber: oldIndex + 1, newLineNumber: newIndex + 1 });
        oldIndex++;
        newIndex++;
      } else {
        const oldLine = oldLines[oldIndex];
        const newLine = newLines[newIndex];

        if (oldIndex + 1 < oldLines.length && oldLines[oldIndex + 1] === newLine) {
          diff.push({ type: 'removed', line: oldLine, oldLineNumber: oldIndex + 1 });
          oldIndex++;
        } else if (newIndex + 1 < newLines.length && oldLine === newLines[newIndex + 1]) {
          diff.push({ type: 'added', line: newLine, newLineNumber: newIndex + 1 });
          newIndex++;
        } else {
          diff.push({ type: 'removed', line: oldLine, oldLineNumber: oldIndex + 1 });
          diff.push({ type: 'added', line: newLine, newLineNumber: newIndex + 1 });
          oldIndex++;
          newIndex++;
        }
      }
    }

    return diff;
  };

  const diff = computeDiff(oldText, newText);

  return (
    <div className={clsx('border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden', className)}>
      <div className="bg-gray-50 dark:bg-gray-800 px-4 py-2 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Diff</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm font-mono">
          <tbody>
            {diff.map((item, index) => (
              <tr
                key={index}
                className={clsx(
                  item.type === 'added' && 'bg-green-50 dark:bg-green-900/20',
                  item.type === 'removed' && 'bg-red-50 dark:bg-red-900/20',
                  item.type === 'unchanged' && 'bg-white dark:bg-gray-900'
                )}
              >
                <td className="px-4 py-1 text-gray-500 dark:text-gray-400 text-right border-r border-gray-200 dark:border-gray-700">
                  {item.oldLineNumber || ''}
                </td>
                <td className="px-4 py-1 text-gray-500 dark:text-gray-400 text-right border-r border-gray-200 dark:border-gray-700">
                  {item.newLineNumber || ''}
                </td>
                <td className="px-4 py-1">
                  {item.type === 'added' && <span className="text-green-600 dark:text-green-400">+ </span>}
                  {item.type === 'removed' && <span className="text-red-600 dark:text-red-400">- </span>}
                  {item.type === 'unchanged' && <span className="text-gray-400">  </span>}
                  <span
                    className={clsx(
                      item.type === 'added' && 'text-green-800 dark:text-green-200',
                      item.type === 'removed' && 'text-red-800 dark:text-red-200',
                      item.type === 'unchanged' && 'text-gray-700 dark:text-gray-300'
                    )}
                  >
                    {item.line || '\u00A0'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};


