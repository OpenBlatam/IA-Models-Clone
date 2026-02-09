/**
 * Memes Page
 * Optimized with code splitting and reusable components
 */

import { Suspense } from 'react';
import { MemesClient } from '@/components/memes/MemesClient';
import { Loading } from '@/components/ui/Loading';

export const dynamic = 'force-dynamic';

/**
 * Memes page with optimized loading states
 */
export default function MemesPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-screen"><Loading size="lg" /></div>}>
      <MemesClient />
    </Suspense>
  );
}
