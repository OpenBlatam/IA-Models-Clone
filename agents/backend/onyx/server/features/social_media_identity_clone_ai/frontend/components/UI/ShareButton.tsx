import { memo, useCallback, useState } from 'react';
import Button from './Button';
import { Share2, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { showToast } from '@/lib/integrations/react-hot-toast';

interface ShareButtonProps {
  title?: string;
  text?: string;
  url?: string;
  className?: string;
  variant?: 'primary' | 'secondary' | 'danger';
  onShare?: () => void;
}

const ShareButton = memo(({
  title,
  text,
  url,
  className = '',
  variant = 'secondary',
  onShare,
}: ShareButtonProps): JSX.Element => {
  const [shared, setShared] = useState(false);

  const handleShare = useCallback(async () => {
    const shareData: ShareData = {
      title: title || document.title,
      text: text || '',
      url: url || window.location.href,
    };

    try {
      if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
        await navigator.share(shareData);
        setShared(true);
        setTimeout(() => setShared(false), 2000);
        if (onShare) {
          onShare();
        }
      } else {
        await navigator.clipboard.writeText(shareData.url || window.location.href);
        showToast.success('Link copied to clipboard!');
        if (onShare) {
          onShare();
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        console.error('Error sharing:', error);
        showToast.error('Failed to share');
      }
    }
  }, [title, text, url, onShare]);

  return (
    <Button
      onClick={handleShare}
      variant={variant}
      className={cn('flex items-center gap-2', className)}
      aria-label="Share"
    >
      {shared ? (
        <>
          <Check className="w-4 h-4" />
          Shared!
        </>
      ) : (
        <>
          <Share2 className="w-4 h-4" />
          Share
        </>
      )}
    </Button>
  );
});

ShareButton.displayName = 'ShareButton';

export default ShareButton;



