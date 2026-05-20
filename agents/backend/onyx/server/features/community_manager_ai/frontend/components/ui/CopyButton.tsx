'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from './Button';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

interface CopyButtonProps {
  text: string;
  label?: string;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  showLabel?: boolean;
}

export const CopyButton = ({
  text,
  label = 'Copiar',
  variant = 'ghost',
  size = 'sm',
  className,
  showLabel = false,
}: CopyButtonProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast.success('Copiado al portapapeles');
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast.error('Error al copiar');
      console.error('Copy error:', err);
    }
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleCopy}
      className={cn('flex items-center gap-2', className)}
      aria-label={label}
    >
      {copied ? (
        <>
          <Check className="h-4 w-4" />
          {showLabel && 'Copiado'}
        </>
      ) : (
        <>
          <Copy className="h-4 w-4" />
          {showLabel && label}
        </>
      )}
    </Button>
  );
};



