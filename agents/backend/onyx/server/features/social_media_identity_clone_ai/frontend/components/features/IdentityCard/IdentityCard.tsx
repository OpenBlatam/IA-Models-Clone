import { memo } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/UI/Card';
import PlatformBadge from '@/components/UI/PlatformBadge';
import StatsGrid from '@/components/Display/StatsGrid';
import { formatDate } from '@/lib/utils';
import type { IdentityProfile } from '@/types';
import { cn } from '@/lib/utils';

interface IdentityCardProps {
  identity: IdentityProfile;
  className?: string;
}

const IdentityCard = memo(({ identity, className = '' }: IdentityCardProps): JSX.Element => {
  const router = useRouter();

  const handleClick = (): void => {
    router.push(`/identities/${identity.profile_id}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <Card
      className={cn('cursor-pointer hover:shadow-lg transition-shadow duration-200', className)}
      role="link"
      tabIndex={0}
      aria-label={`View identity ${identity.username}`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
    >
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold">{identity.username}</h3>
          <div className="flex gap-2">
            {identity.tiktok_profile && <PlatformBadge platform="tiktok" />}
            {identity.instagram_profile && <PlatformBadge platform="instagram" />}
            {identity.youtube_profile && <PlatformBadge platform="youtube" />}
          </div>
        </div>

        {identity.display_name && <p className="text-gray-600">{identity.display_name}</p>}

        {identity.bio && <p className="text-sm text-gray-600 line-clamp-2">{identity.bio}</p>}

        <StatsGrid
          stats={[
            { label: 'Videos', value: identity.total_videos },
            { label: 'Posts', value: identity.total_posts },
            { label: 'Comments', value: identity.total_comments },
          ]}
          columns={3}
          className="pt-4 border-t"
        />

        {identity.content_analysis.tone && (
          <div className="pt-2">
            <span className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-xs">
              {identity.content_analysis.tone}
            </span>
          </div>
        )}

        <p className="text-xs text-gray-500">{formatDate(identity.created_at)}</p>
      </div>
    </Card>
  );
});

IdentityCard.displayName = 'IdentityCard';

export default IdentityCard;



