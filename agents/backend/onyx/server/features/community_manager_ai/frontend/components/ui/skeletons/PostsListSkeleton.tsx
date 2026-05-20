/**
 * Posts List Skeleton Component
 * Loading skeleton for posts list
 */

import { PostSkeleton } from './PostSkeleton';

interface PostsListSkeletonProps {
  count?: number;
}

/**
 * Skeleton component for posts list
 */
export const PostsListSkeleton = ({ count = 5 }: PostsListSkeletonProps) => {
  return (
    <div className="grid gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <PostSkeleton key={i} />
      ))}
    </div>
  );
};


