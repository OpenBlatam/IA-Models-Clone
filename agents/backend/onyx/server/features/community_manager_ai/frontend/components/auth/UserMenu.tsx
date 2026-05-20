'use client';

import { LogOut, User as UserIcon } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { useTranslations } from 'next-intl';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { cn } from '@/lib/utils';

export const UserMenu = () => {
  const { user, signOut } = useAuth();
  const t = useTranslations('auth');

  const handleSignOut = () => {
    signOut();
  };

  if (!user) return null;

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className="flex items-center gap-2 rounded-lg p-2 text-gray-600 hover:bg-gray-100 transition-colors"
          aria-label="Menú de usuario"
          tabIndex={0}
        >
          {user?.image ? (
            <img
              src={user.image}
              alt={user.name || 'Usuario'}
              className="h-8 w-8 rounded-full"
            />
          ) : (
            <UserIcon className="h-5 w-5" />
          )}
          <span className="hidden sm:block text-sm font-medium">{user?.name || user?.email}</span>
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className={cn(
            'min-w-[200px] rounded-lg border border-gray-200 bg-white p-1 shadow-lg z-50',
            'animate-in fade-in-0 zoom-in-95'
          )}
          sideOffset={5}
          align="end"
        >
          <DropdownMenu.Item className="px-3 py-2 text-sm text-gray-700 outline-none cursor-default">
            <div className="flex flex-col">
              <span className="font-medium">{user?.name}</span>
              <span className="text-xs text-gray-500">{user?.email}</span>
            </div>
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="my-1 h-px bg-gray-200" />
          <DropdownMenu.Item asChild>
            <button
              type="button"
              onClick={handleSignOut}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors outline-none"
              aria-label={t('signOut')}
              tabIndex={0}
            >
              <LogOut className="h-4 w-4" />
              {t('signOut')}
            </button>
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
};
