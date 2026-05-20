/**
 * Posts Page
 * Optimized with code splitting and reusable components
 */

import { Suspense } from 'react';
import { PostsClient } from '@/components/posts/PostsClient';
import { Loading } from '@/components/ui/Loading';

export const dynamic = 'force-dynamic';

/**
 * Posts page with optimized loading states
 */
export default function PostsPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-screen"><Loading size="lg" /></div>}>
      <PostsClient />
    </Suspense>
  );
}
