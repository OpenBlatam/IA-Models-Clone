/**
 * Dashboard Page
 * Optimized with code splitting and Suspense boundaries
 */

import { Suspense } from 'react';
import { DashboardClient } from '@/components/dashboard/DashboardClient';
import { Loading } from '@/components/ui/Loading';
import { useTranslations } from 'next-intl';

export const dynamic = 'force-dynamic';

/**
 * Dashboard page with optimized loading states
 */
export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-screen"><Loading size="lg" /></div>}>
      <DashboardClient />
    </Suspense>
  );
}

