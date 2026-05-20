'use client';

import { ReactNode } from 'react';
import { FiCopy, FiCheck } from 'react-icons/fi';
import { useCopyToClipboard } from '@/hooks';
import { cn } from '@/utils/classNames';
import { Button } from './Button';

interface CopyButtonProps {
  text: string;
  children?: ReactNode;
  className?: string;
  variant?: 'default' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg';
}

export function CopyButton({
  text,
  children,
  className,
  variant = 'ghost',
  size = 'sm',
}: CopyButtonProps) {
  const { copy, copied } = useCopyToClipboard();

  return (
    <Button
      variant={variant}
      size={size}
      onClick={() => copy(text)}
      className={cn(className)}
      aria-label={copied ? 'Copiado' : 'Copiar'}
    >
      {children || (
        <>
          {copied ? (
            <>
              <FiCheck size={16} />
              <span>Copiado</span>
            </>
          ) : (
            <>
              <FiCopy size={16} />
              <span>Copiar</span>
            </>
          )}
        </>
      )}
    </Button>
  );
}

