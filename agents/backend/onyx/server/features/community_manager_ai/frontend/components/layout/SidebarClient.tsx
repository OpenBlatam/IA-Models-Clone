/**
 * Sidebar Client Component
 * Client-side sidebar with navigation and responsive behavior
 */

'use client';

import { usePathname } from '@/i18n/routing';
import { Link } from '@/i18n/routing';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/lib/store';
import {
  LayoutDashboard,
  FileText,
  Image,
  Calendar,
  Share2,
  BarChart3,
  File,
  Settings,
  CreditCard,
  Menu,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useMediaQuery } from '@/hooks/useMediaQuery';

const navigation = [
  { key: 'dashboard', href: '/dashboard', icon: LayoutDashboard },
  { key: 'posts', href: '/posts', icon: FileText },
  { key: 'memes', href: '/memes', icon: Image },
  { key: 'calendar', href: '/calendar', icon: Calendar },
  { key: 'platforms', href: '/platforms', icon: Share2 },
  { key: 'analytics', href: '/analytics', icon: BarChart3 },
  { key: 'templates', href: '/templates', icon: File },
  { key: 'pricing', href: '/pricing', icon: CreditCard },
  { key: 'subscription', href: '/subscription', icon: CreditCard },
  { key: 'settings', href: '/settings', icon: Settings },
] as const;

/**
 * Sidebar navigation component with responsive behavior
 */
export const SidebarClient = () => {
  const pathname = usePathname();
  const t = useTranslations('nav');
  const { sidebarOpen, setSidebarOpen } = useAppStore();
  const isMobile = useMediaQuery('(max-width: 768px)');

  // Close sidebar on mobile when route changes
  const handleLinkClick = () => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed left-0 top-0 z-50 flex h-screen w-64 flex-col bg-gray-900 dark:bg-gray-950 text-white border-r border-gray-800 dark:border-gray-900 transition-transform duration-300 ease-in-out',
          isMobile && !sidebarOpen && '-translate-x-full',
          isMobile && sidebarOpen && 'translate-x-0',
          'lg:relative lg:translate-x-0'
        )}
        aria-label="Navegación principal"
      >
        {/* Header with mobile toggle */}
        <div className="flex h-16 items-center justify-between border-b border-gray-800 dark:border-gray-900 px-4">
          <h1 className="text-xl font-bold">Community Manager AI</h1>
          {isMobile && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(false)}
              className="text-white hover:bg-gray-800"
              aria-label="Cerrar menú"
            >
              <X className="h-5 w-5" />
            </Button>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 overflow-y-auto px-2 py-4" aria-label="Navegación principal">
          {navigation.map((item) => {
            const isActive = pathname === item.href || pathname?.startsWith(item.href + '/');
            const Icon = item.icon;

            return (
              <Link
                key={item.key}
                href={item.href}
                onClick={handleLinkClick}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-gray-900',
                  isActive
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-300 dark:text-gray-400 hover:bg-gray-800 dark:hover:bg-gray-800 hover:text-white'
                )}
                aria-current={isActive ? 'page' : undefined}
                aria-label={t(item.key)}
              >
                <Icon className="h-5 w-5 shrink-0" aria-hidden="true" />
                <span>{t(item.key)}</span>
              </Link>
            );
          })}
        </nav>
      </aside>
    </>
  );
};


