'use client';

import { memo } from 'react';
import { Share2 } from 'lucide-react';
import { useShare } from '@/lib/hooks';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface ShareButtonProps {
  title?: string;
  text?: string;
  url?: string;
  files?: File[];
  className?: string;
  variant?: 'default' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  showText?: boolean;
  onShare?: () => void;
  onError?: (error: Error) => void;
}

const ShareButton = memo(
  ({
    title,
    text,
    url,
    files,
    className,
    variant = 'ghost',
    size = 'icon',
    showText = false,
    onShare,
    onError,
  }: ShareButtonProps): JSX.Element => {
    const { share, isSupported, error } = useShare();

    const handleShare = async (): Promise<void> => {
      const success = await share({
        title,
        text,
        url: url || window.location.href,
        files,
      });

      if (success) {
        onShare?.();
      } else if (error) {
        onError?.(error);
      }
    };

    if (!isSupported) {
      return null;
    }

    return (
      <Button
        variant={variant}
        size={size}
        onClick={handleShare}
        className={cn(className)}
        aria-label="Share"
      >
        <Share2 className="w-4 h-4" aria-hidden="true" />
        {showText && <span className="ml-2">Share</span>}
      </Button>
    );
  }
);

ShareButton.displayName = 'ShareButton';

export default ShareButton;

