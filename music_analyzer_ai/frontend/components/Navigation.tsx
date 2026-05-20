/**
 * Navigation component for the application.
 * Provides main navigation links between different sections.
 * Enhanced with better accessibility and responsive design.
 */

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Music, Bot, Home } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ROUTES } from '@/lib/constants';
import { useMemo } from 'react';

/**
 * Navigation item interface.
 */
interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  description?: string;
}

/**
 * Navigation items configuration.
 */
const NAV_ITEMS: NavItem[] = [
  {
    href: ROUTES.HOME,
    label: 'Inicio',
    icon: Home,
    description: 'Página principal',
  },
  {
    href: ROUTES.MUSIC,
    label: 'Music AI',
    icon: Music,
    description: 'Analizador de música con IA',
  },
  {
    href: ROUTES.ROBOT,
    label: 'Robot AI',
    icon: Bot,
    description: 'Control de robot con IA',
  },
] as const;

/**
 * Main navigation component.
 * Displays navigation links with active state highlighting.
 * Optimized for accessibility and responsive design.
 *
 * @returns Navigation component
 */
export function Navigation() {
  const pathname = usePathname();

  /**
   * Checks if a route is active.
   */
  const isActive = useMemo(
    () => (href: string) => {
      if (href === ROUTES.HOME) {
        return pathname === ROUTES.HOME;
      }
      return pathname.startsWith(href);
    },
    [pathname]
  );

  return (
    <nav
      className="bg-white/10 backdrop-blur-lg border-b border-white/20 sticky top-0 z-50"
      role="navigation"
      aria-label="Navegación principal"
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link
            href={ROUTES.HOME}
            className="text-xl font-bold text-white hover:text-purple-300 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900 rounded"
            aria-label="Blatam Academy - Ir al inicio"
          >
            Blatam Academy
          </Link>
          <div className="flex items-center gap-2 sm:gap-4">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg transition-colors',
                    'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900',
                    active
                      ? 'bg-white/20 text-white'
                      : 'text-gray-300 hover:bg-white/10 hover:text-white'
                  )}
                  aria-label={item.description || item.label}
                  aria-current={active ? 'page' : undefined}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" aria-hidden="true" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
