'use client';

import { ReactNode } from 'react';
import { Spinner } from './Spinner';

interface LoadingProps {
  children?: ReactNode;
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  fullScreen?: boolean;
}

export function Loading({
  children,
  size = 'md',
  text = 'Cargando...',
  fullScreen = false,
}: LoadingProps) {
  const content = (
    <div className="flex flex-col items-center justify-center gap-4">
      <Spinner size={size} />
      {text && (
        <p className="text-sm text-gray-600 dark:text-gray-400">{text}</p>
      )}
      {children}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white dark:bg-gray-900 bg-opacity-75 dark:bg-opacity-75">
        {content}
      </div>
    );
  }

  return <div className="py-12">{content}</div>;
}

