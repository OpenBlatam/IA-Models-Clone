/**
 * Memes List Component
 * Server Component for displaying list of memes
 */

import { Suspense } from 'react';
import { MemeCard } from './MemeCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { Loading } from '@/components/ui/Loading';
import { MemesGridSkeleton } from './MemesGridSkeleton';
import { Meme } from '@/types';
import { Image as ImageIcon } from 'lucide-react';

interface MemesListProps {
  memes: Meme[];
  isLoading?: boolean;
  onDelete: (memeId: string) => void;
  emptyTitle?: string;
  emptyDescription?: string;
  emptyActionLabel?: string;
  onEmptyAction?: () => void;
}

/**
 * Memes list component with loading and empty states
 */
export const MemesList = ({
  memes,
  isLoading,
  onDelete,
  emptyTitle = 'No hay memes',
  emptyDescription = 'Sube tu primer meme para comenzar a gestionar tu biblioteca de contenido.',
  emptyActionLabel = 'Subir Meme',
  onEmptyAction,
}: MemesListProps) => {
  if (isLoading) {
    return <MemesGridSkeleton />;
  }

  if (memes.length === 0) {
    return (
      <EmptyState
        icon={ImageIcon}
        title={emptyTitle}
        description={emptyDescription}
        actionLabel={emptyActionLabel}
        onAction={onEmptyAction}
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      <Suspense fallback={<MemesGridSkeleton />}>
        {memes.map((meme) => (
          <MemeCard
            key={meme.meme_id}
            meme={meme}
            onDelete={onDelete}
            isLoading={isLoading}
          />
        ))}
      </Suspense>
    </div>
  );
};


