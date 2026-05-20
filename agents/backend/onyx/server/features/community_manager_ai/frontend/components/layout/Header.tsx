/**
 * Header Component
 * Application header with navigation and user actions
 */

'use client';

import { useTranslations } from 'next-intl';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { UserMenu } from '@/components/auth/UserMenu';
import { NotificationBell } from '@/components/ui/NotificationBell';
import { CommandPalette } from '@/components/ui/CommandPalette';
import { SidebarToggle } from './SidebarToggle';

/**
 * Header component with responsive design
 */
export const Header = () => {
  const t = useTranslations('common');

  return (
    <header
      className="sticky top-0 z-10 flex h-16 items-center justify-between border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 sm:px-6 shadow-sm backdrop-blur-sm bg-white/95 dark:bg-gray-900/95"
      role="banner"
    >
      <div className="flex flex-1 items-center gap-2 sm:gap-4">
        <SidebarToggle />
        <CommandPalette />
      </div>
      <div className="flex items-center gap-2 sm:gap-4">
        <ThemeToggle />
        <LanguageSelector />
        <NotificationBell />
        <UserMenu />
      </div>
    </header>
  );
};

