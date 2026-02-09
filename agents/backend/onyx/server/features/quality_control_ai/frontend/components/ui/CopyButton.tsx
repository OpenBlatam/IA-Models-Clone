'use client';

import { memo } from 'react';
import { Copy, Check } from 'lucide-react';
import { useClipboard } from '@/lib/hooks';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface CopyButtonProps {
  text: string;
  className?: string;
  variant?: 'default' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  showText?: boolean;
  copiedText?: string;
  onCopy?: () => void;
}

const CopyButton = memo(
  ({
    text,
    className,
    variant = 'ghost',
    size = 'icon',
    showText = false,
    copiedText = 'Copied!',
    onCopy,
  }: CopyButtonProps): JSX.Element => {
    const { copy, copied } = useClipboard({
      onSuccess: onCopy,
    });

    const handleClick = async (): Promise<void> => {
      await copy(text);
    };

    return (
      <Button
        variant={variant}
        size={size}
        onClick={handleClick}
        className={cn(className)}
        aria-label={copied ? 'Copied' : 'Copy to clipboard'}
      >
        {copied ? (
          <>
            <Check className="w-4 h-4" aria-hidden="true" />
            {showText && <span className="ml-2">{copiedText}</span>}
          </>
        ) : (
          <>
            <Copy className="w-4 h-4" aria-hidden="true" />
            {showText && <span className="ml-2">Copy</span>}
          </>
        )}
      </Button>
    );
  }
);

CopyButton.displayName = 'CopyButton';

export default CopyButton;
