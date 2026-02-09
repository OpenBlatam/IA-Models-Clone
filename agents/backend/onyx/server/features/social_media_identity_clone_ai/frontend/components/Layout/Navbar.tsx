'use client';

import { memo, useCallback } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';

interface NavItem {
  href: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { href: '/', label: 'Home' },
  { href: '/extract-profile', label: 'Extract Profile' },
  { href: '/build-identity', label: 'Build Identity' },
  { href: '/generate-content', label: 'Generate Content' },
  { href: '/identities', label: 'Identities' },
  { href: '/dashboard', label: 'Dashboard' },
  { href: '/analytics', label: 'Analytics' },
  { href: '/search', label: 'Search' },
  { href: '/templates', label: 'Templates' },
  { href: '/tasks', label: 'Tasks' },
  { href: '/notifications', label: 'Notifications' },
  { href: '/recommendations', label: 'Recommendations' },
  { href: '/scheduler', label: 'Scheduler' },
  { href: '/ab-testing', label: 'A/B Testing' },
  { href: '/backups', label: 'Backups' },
  { href: '/collaboration', label: 'Collaboration' },
  { href: '/ml', label: 'ML' },
  { href: '/webhooks', label: 'Webhooks' },
  { href: '/plugins', label: 'Plugins' },
  { href: '/batch', label: 'Batch' },
  { href: '/alerts', label: 'Alerts' },
  { href: '/validate', label: 'Validate' },
  { href: '/health', label: 'Health' },
];

interface NavLinkProps {
  item: NavItem;
  isActive: boolean;
}

const NavLink = memo(({ item, isActive }: NavLinkProps): JSX.Element => {
  const router = useRouter();

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLAnchorElement>): void => {
      if (e.key !== 'Enter' && e.key !== ' ') {
        return;
      }
      e.preventDefault();
      router.push(item.href);
    },
    [item.href, router]
  );

  const linkClasses = cn(
    'px-3 py-2 rounded-md text-sm font-medium transition-colors',
    isActive ? 'bg-primary-100 text-primary-700' : 'text-gray-700 hover:bg-gray-100'
  );

  return (
    <Link
      href={item.href}
      className={linkClasses}
      tabIndex={0}
      aria-label={item.label}
      aria-current={isActive ? 'page' : undefined}
      onKeyDown={handleKeyDown}
    >
      {item.label}
    </Link>
  );
});

NavLink.displayName = 'NavLink';

const Navbar = memo((): JSX.Element => {
  const pathname = usePathname();

  return (
    <nav className="bg-white shadow-md" role="navigation" aria-label="Main navigation">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link
            href="/"
            className="text-xl font-bold text-primary-600 hover:text-primary-700 transition-colors"
            tabIndex={0}
            aria-label="Home - Identity Clone AI"
          >
            Identity Clone AI
          </Link>
          <div className="flex space-x-4" role="list">
            {NAV_ITEMS.map((item) => (
              <NavLink key={item.href} item={item} isActive={pathname === item.href} />
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
});

Navbar.displayName = 'Navbar';

export default Navbar;
