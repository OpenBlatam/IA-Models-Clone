'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface BlockquoteProps {
  children: ReactNode;
  author?: string;
  cite?: string;
  className?: string;
}

export const Blockquote = ({ children, author, cite, className }: BlockquoteProps) => {
  return (
    <blockquote
      className={cn(
        'border-l-4 border-primary-500 dark:border-primary-400 pl-4 py-2 my-4',
        'italic text-gray-700 dark:text-gray-300',
        className
      )}
      cite={cite}
    >
      <p>{children}</p>
      {author && (
        <footer className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          — {author}
        </footer>
      )}
    </blockquote>
  );
};



