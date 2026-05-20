/**
 * Skip link component for accessibility.
 * Allows keyboard users to skip to main content.
 */

import Link from 'next/link';
import { cn } from '@/lib/utils';

interface SkipLinkProps {
  href?: string;
  className?: string;
}

/**
 * Skip link component.
 * Provides keyboard navigation to skip to main content.
 * Important for accessibility compliance.
 *
 * @param props - Component props
 * @returns Skip link component
 */
export function SkipLink({
  href = '#main-content',
  className,
}: SkipLinkProps) {
  return (
    <Link
      href={href}
      className={cn(
        'sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4',
        'focus:z-50 focus:px-4 focus:py-2',
        'focus:bg-purple-600 focus:text-white focus:rounded-lg',
        'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2',
        className
      )}
      aria-label="Saltar al contenido principal"
    >
      Saltar al contenido
    </Link>
  );
}

