'use client';

import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { clsx } from 'clsx';
import { useCopyToClipboard } from '@/hooks';

interface CodeBlockProps {
  code: string;
  language?: string;
  showLineNumbers?: boolean;
  className?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language = 'javascript',
  showLineNumbers = false,
  className,
}) => {
  const { copy, copied } = useCopyToClipboard();
  const [hovered, setHovered] = useState(false);

  const handleCopy = () => {
    copy(code);
  };

  const lines = code.split('\n');

  return (
    <div
      className={clsx(
        'relative rounded-lg bg-gray-900 dark:bg-gray-950 overflow-hidden',
        className
      )}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 dark:bg-gray-900 border-b border-gray-700">
        <span className="text-xs text-gray-400 uppercase">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center space-x-2 px-2 py-1 text-xs text-gray-400 hover:text-white transition-colors"
        >
          {copied ? (
            <>
              <Check className="h-4 w-4" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-4 w-4" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="overflow-x-auto">
        <pre className="p-4 text-sm">
          <code className="text-gray-100">
            {showLineNumbers ? (
              <div className="flex">
                <div className="pr-4 text-gray-500 select-none">
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
    </div>
  );
};


