import { memo } from 'react';
import Card from '@/components/UI/Card';
import PlatformBadge from '@/components/UI/PlatformBadge';
import { formatDate } from '@/lib/utils';
import { Platform } from '@/types';
import { cn } from '@/lib/utils';

interface ContentCardProps {
  title?: string;
  content: string;
  platform: Platform;
  createdAt: string;
  hashtags?: string[];
  className?: string;
  onClick?: () => void;
}

const ContentCard = memo(({
  title,
  content,
  platform,
  createdAt,
  hashtags = [],
  className = '',
  onClick,
}: ContentCardProps): JSX.Element => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (onClick && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <Card
      className={cn(onClick && 'cursor-pointer hover:shadow-lg transition-shadow', className)}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <PlatformBadge platform={platform} />
          <span className="text-xs text-gray-500">{formatDate(createdAt)}</span>
        </div>

        {title && <h3 className="font-semibold text-lg">{title}</h3>}

        <div className="bg-gray-50 p-4 rounded-lg whitespace-pre-wrap text-sm">{content}</div>

        {hashtags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {hashtags.map((tag, idx) => (
              <span key={idx} className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-xs">
                #{tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
});

ContentCard.displayName = 'ContentCard';

export default ContentCard;



