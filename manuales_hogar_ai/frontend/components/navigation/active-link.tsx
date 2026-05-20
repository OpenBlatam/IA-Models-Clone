'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Button } from '../ui/button';
import { cn } from '@/lib/utils/cn';
import type { ActiveLinkProps } from '@/lib/types/components';

export const ActiveLink = ({ href, label, icon: Icon, exact = false }: ActiveLinkProps): JSX.Element => {
  const pathname = usePathname();
  const isActive = exact ? pathname === href : pathname.startsWith(href);

  return (
    <Link href={href}>
      <Button
        variant={isActive ? 'default' : 'ghost'}
        className={cn(
          'flex items-center space-x-2',
          isActive && 'bg-indigo-600 text-white hover:bg-indigo-700'
        )}
        tabIndex={0}
        aria-label={label}
        aria-current={isActive ? 'page' : undefined}
      >
        <Icon className="h-5 w-5" />
        <span>{label}</span>
      </Button>
    </Link>
  );
};

