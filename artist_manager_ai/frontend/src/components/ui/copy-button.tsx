'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface CopyButtonProps {
  text: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'ghost';
}

const CopyButton = ({ text, className, size = 'sm', variant = 'ghost' }: CopyButtonProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleCopy}
      className={cn(className)}
      aria-label={copied ? 'Copiado' : 'Copiar'}
    >
      {copied ? (
        <>
          <Check className="w-4 h-4 mr-2" />
          Copiado
        </>
      ) : (
        <>
          <Copy className="w-4 h-4 mr-2" />
          Copiar
        </>
      )}
    </Button>
  );
};

export { CopyButton };

