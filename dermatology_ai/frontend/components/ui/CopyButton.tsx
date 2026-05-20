'use client';

import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from './Button';
import { clsx } from 'clsx';
import toast from 'react-hot-toast';

interface CopyButtonProps {
  text: string;
  label?: string;
  variant?: 'button' | 'icon';
  className?: string;
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  text,
  label = 'Copy',
  variant = 'button',
  className,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast.success('Copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Error copying');
    }
  };

  if (variant === 'icon') {
    return (
      <button
        onClick={handleCopy}
        className={clsx(
          'p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors',
          className
        )}
        aria-label="Copy"
      >
        {copied ? (
          <Check className="h-4 w-4 text-green-600 dark:text-green-400" />
        ) : (
          <Copy className="h-4 w-4 text-gray-600 dark:text-gray-400" />
        )}
      </button>
    );
  }

  return (
    <Button
      onClick={handleCopy}
      variant="outline"
      size="sm"
      className={clsx('flex items-center space-x-2', className)}
    >
      {copied ? (
        <>
          <Check className="h-4 w-4" />
          <span>Copied</span>
        </>
      ) : (
        <>
          <Copy className="h-4 w-4" />
          <span>{label}</span>
        </>
      )}
    </Button>
  );
};


