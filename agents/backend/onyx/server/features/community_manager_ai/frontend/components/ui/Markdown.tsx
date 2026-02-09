'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface MarkdownProps {
  content: string;
  className?: string;
}

export const Markdown = ({ content, className }: MarkdownProps) => {
  const parseMarkdown = (text: string): ReactNode[] => {
    const lines = text.split('\n');
    const elements: ReactNode[] = [];

    lines.forEach((line, index) => {
      if (line.startsWith('# ')) {
        elements.push(
          <h1 key={index} className="text-3xl font-bold mb-4 text-gray-900 dark:text-gray-100">
            {line.substring(2)}
          </h1>
        );
      } else if (line.startsWith('## ')) {
        elements.push(
          <h2 key={index} className="text-2xl font-semibold mb-3 text-gray-900 dark:text-gray-100">
            {line.substring(3)}
          </h2>
        );
      } else if (line.startsWith('### ')) {
        elements.push(
          <h3 key={index} className="text-xl font-medium mb-2 text-gray-900 dark:text-gray-100">
            {line.substring(4)}
          </h3>
        );
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        elements.push(
          <li key={index} className="ml-4 list-disc text-gray-700 dark:text-gray-300">
            {line.substring(2)}
          </li>
        );
      } else if (line.trim() === '') {
        elements.push(<br key={index} />);
      } else {
        elements.push(
          <p key={index} className="mb-2 text-gray-700 dark:text-gray-300">
            {line}
          </p>
        );
      }
    });

    return elements;
  };

  return (
    <div className={cn('prose prose-gray dark:prose-invert max-w-none', className)}>
      {parseMarkdown(content)}
    </div>
  );
};



