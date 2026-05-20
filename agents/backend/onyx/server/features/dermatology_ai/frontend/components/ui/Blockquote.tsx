'use client';

import React from 'react';
import { clsx } from 'clsx';

interface BlockquoteProps {
  children: React.ReactNode;
  author?: string;
  className?: string;
}

export const Blockquote: React.FC<BlockquoteProps> = ({
  children,
  author,
  className,
}) => {
  return (
    <blockquote
      className={clsx(
        'border-l-4 border-primary-500 pl-4 py-2 italic text-gray-700 dark:text-gray-300',
        className
      )}
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
