/**
 * Copy to clipboard button component
 */

import React, { useState } from 'react';
import { Button } from './Button';
import { Copy, Check } from 'lucide-react';
import toast from 'react-hot-toast';

export interface CopyButtonProps {
  text: string;
  label?: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const CopyButton: React.FC<CopyButtonProps> = ({
  text,
  label = 'Copiar',
  variant = 'outline',
  size = 'sm',
  className,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!text) {
      toast.error('No hay texto para copiar');
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast.success('Copiado al portapapeles');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Error al copiar');
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleCopy();
    }
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleCopy}
      onKeyDown={handleKeyDown}
      className={className}
      aria-label={`${label}: ${text}`}
      tabIndex={0}
    >
      {copied ? (
        <>
          <Check className="h-4 w-4 mr-2" aria-hidden="true" />
          Copiado
        </>
      ) : (
        <>
          <Copy className="h-4 w-4 mr-2" aria-hidden="true" />
          {label}
        </>
      )}
    </Button>
  );
};

export { CopyButton };




