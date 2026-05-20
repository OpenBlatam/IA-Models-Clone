'use client';

import { CopyButton } from './CopyButton';
import { cn } from '@/lib/utils';

interface CodeBlockProps {
  code: string;
  language?: string;
  className?: string;
  showCopy?: boolean;
}

export const CodeBlock = ({
  code,
  language,
  className,
  showCopy = true,
}: CodeBlockProps) => {
  return (
    <div className={cn('relative', className)}>
      {showCopy && (
        <div className="absolute right-2 top-2 z-10">
          <CopyButton text={code} variant="ghost" size="sm" />
        </div>
      )}
      <pre className="overflow-x-auto rounded-lg bg-gray-900 dark:bg-gray-950 p-4 text-sm text-gray-100">
        <code className={language ? `language-${language}` : ''}>{code}</code>
      </pre>
    </div>
  );
};



