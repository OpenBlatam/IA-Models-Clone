import { memo, useState, useCallback } from 'react';
import Button from './Button';
import { cn } from '@/lib/utils';
import { Copy, Check } from 'lucide-react';

interface ClipboardProps {
  text: string;
  label?: string;
  className?: string;
  variant?: 'button' | 'icon';
  onCopy?: () => void;
}

const Clipboard = memo(({
  text,
  label = 'Copy',
  className = '',
  variant = 'button',
  onCopy,
}: ClipboardProps): JSX.Element => {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (onCopy) {
        onCopy();
      }
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  }, [text, onCopy]);

  if (variant === 'icon') {
    return (
      <button
        onClick={handleCopy}
        className={cn('p-2 rounded hover:bg-gray-100 transition-colors', className)}
        aria-label={label}
        title={label}
      >
        {copied ? (
          <Check className="w-4 h-4 text-green-600" />
        ) : (
          <Copy className="w-4 h-4 text-gray-600" />
        )}
      </button>
    );
  }

  return (
    <Button
      onClick={handleCopy}
      variant="secondary"
      className={cn('flex items-center gap-2', className)}
      aria-label={label}
    >
      {copied ? (
        <>
          <Check className="w-4 h-4" />
          Copied!
        </>
      ) : (
        <>
          <Copy className="w-4 h-4" />
          {label}
        </>
      )}
    </Button>
  );
});

Clipboard.displayName = 'Clipboard';

export default Clipboard;



