import { useState } from 'react';
import Button from './Button';
import { cn } from '@/lib/utils';

interface CopyButtonProps {
  text: string;
  label?: string;
  className?: string;
  variant?: 'primary' | 'secondary' | 'danger';
}

const COPY_SUCCESS_DURATION = 2000;

const CopyButton = ({
  text,
  label = 'Copy',
  className = '',
  variant = 'secondary',
}: CopyButtonProps): JSX.Element => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      
      setTimeout(() => {
        setCopied(false);
      }, COPY_SUCCESS_DURATION);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleCopy();
    }
  };

  return (
    <Button
      variant={variant}
      onClick={handleCopy}
      onKeyDown={handleKeyDown}
      className={cn('text-sm', className)}
      aria-label={copied ? 'Copied!' : `Copy ${label}`}
    >
      {copied ? 'Copied!' : label}
    </Button>
  );
};

export default CopyButton;



