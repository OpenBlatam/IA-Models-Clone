/**
 * Posts List Component
 * Server Component for displaying list of posts
 */

import { Suspense } from 'react';
import { PostCard } from './PostCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { Loading } from '@/components/ui/Loading';
import { Post } from '@/types';
import { FileText } from 'lucide-react';

interface PostsListProps {
  posts: Post[];
  isLoading?: boolean;
  onEdit: (post: Post) => void;
  onDelete: (postId: string) => void;
  onPublish?: (postId: string) => void;
  emptyTitle?: string;
  emptyDescription?: string;
  emptyActionLabel?: string;
  onEmptyAction?: () => void;
}

/**
 * Posts list component with loading and empty states
 */
export const PostsList = ({
  posts,
  isLoading,
  onEdit,
  onDelete,
  onPublish,
  emptyTitle = 'No hay posts',
  emptyDescription = 'Comienza creando tu primer post para gestionar tus publicaciones en redes sociales.',
  emptyActionLabel = 'Crear Post',
  onEmptyAction,
}: PostsListProps) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loading size="lg" text="Cargando posts..." />
      </div>
    );
  }

  if (posts.length === 0) {
    return (
      <EmptyState
        icon={FileText}
        title={emptyTitle}
        description={emptyDescription}
        actionLabel={emptyActionLabel}
        onAction={onEmptyAction}
      />
    );
  }

  return (
    <div className="grid gap-4">
      <Suspense fallback={<Loading size="lg" />}>
        {posts.map((post) => (
          <PostCard
            key={post.post_id}
            post={post}
            onEdit={onEdit}
            onDelete={onDelete}
            onPublish={onPublish}
            isLoading={isLoading}
          />
        ))}
      </Suspense>
    </div>
  );
};


