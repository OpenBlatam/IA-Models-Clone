'use client';

import { useState } from 'react';
import { FiCopy, FiCheck } from 'react-icons/fi';
import { useCopyToClipboard } from '@/hooks';
import { cn } from '@/utils/classNames';

interface CodeBlockProps {
  code: string;
  language?: string;
  showLineNumbers?: boolean;
  className?: string;
}

export function CodeBlock({
  code,
  language = 'text',
  showLineNumbers = false,
  className,
}: CodeBlockProps) {
  const { copy, copied } = useCopyToClipboard();
  const lines = code.split('\n');

  return (
    <div className={cn('relative group', className)}>
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 dark:bg-gray-900 border-b border-gray-700 rounded-t-lg">
        <span className="text-xs text-gray-400 uppercase">{language}</span>
        <button
          onClick={() => copy(code)}
          className="flex items-center gap-2 px-2 py-1 text-xs text-gray-400 hover:text-white transition-colors"
          aria-label="Copiar código"
        >
          {copied ? (
            <>
              <FiCheck size={14} />
              <span>Copiado</span>
            </>
          ) : (
            <>
              <FiCopy size={14} />
              <span>Copiar</span>
            </>
          )}
        </button>
      </div>
      <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded-b-lg overflow-x-auto">
        <code className="text-sm font-mono">
          {showLineNumbers ? (
            <div className="flex">
              <div className="select-none text-gray-600 dark:text-gray-500 pr-4 text-right">
                {lines.map((_, i) => (
                  <div key={i}>{i + 1}</div>
                ))}
              </div>
              <div className="flex-1">
                {lines.map((line, i) => (
                  <div key={i}>{line || '\u00A0'}</div>
                ))}
              </div>
            </div>
          ) : (
            code
          )}
        </code>
      </pre>
    </div>
  );
}

