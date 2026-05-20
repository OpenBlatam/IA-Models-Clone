'use client';

import { Share2 } from 'lucide-react';
import { Button } from './Button';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

interface ShareButtonProps {
  url?: string;
  title?: string;
  text?: string;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  label?: string;
}

export const ShareButton = ({
  url = typeof window !== 'undefined' ? window.location.href : '',
  title,
  text,
  variant = 'ghost',
  size = 'sm',
  className,
  label = 'Compartir',
}: ShareButtonProps) => {
  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: title || document.title,
          text: text || '',
          url: url,
        });
        toast.success('Compartido exitosamente');
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          handleFallbackShare();
        }
      }
    } else {
      handleFallbackShare();
    }
  };

  const handleFallbackShare = () => {
    navigator.clipboard.writeText(url);
    toast.success('URL copiada al portapapeles');
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleShare}
      className={cn('flex items-center gap-2', className)}
      aria-label={label}
    >
      <Share2 className="h-4 w-4" />
      {label}
    </Button>
  );
};



