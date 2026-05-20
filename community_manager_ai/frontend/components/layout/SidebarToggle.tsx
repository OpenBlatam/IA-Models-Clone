/**
 * Sidebar Toggle Component
 * Mobile sidebar toggle button
 */

'use client';

import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useAppStore } from '@/lib/store';
import { useMediaQuery } from '@/hooks/useMediaQuery';

/**
 * Sidebar toggle button for mobile
 */
export const SidebarToggle = () => {
  const { sidebarOpen, setSidebarOpen } = useAppStore();
  const isMobile = useMediaQuery('(max-width: 768px)');

  if (!isMobile) return null;

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setSidebarOpen(!sidebarOpen)}
      className="lg:hidden"
      aria-label="Toggle sidebar"
      aria-expanded={sidebarOpen}
    >
      <Menu className="h-5 w-5" />
    </Button>
  );
};


