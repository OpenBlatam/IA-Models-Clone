/**
 * Conditional layout wrapper.
 * Conditionally renders Navigation and Footer based on the current route.
 */

'use client';

import { usePathname } from 'next/navigation';
import { Navigation } from './Navigation';
import { Footer } from './Footer';

/**
 * Conditional layout component.
 * Hides Navigation and Footer on the homepage.
 *
 * @returns Conditional layout wrapper
 */
export function ConditionalLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isHomePage = pathname === '/';
  const isDashboard = pathname?.startsWith('/dashboard');

  return (
    <>
      {!isHomePage && !isDashboard && <Navigation />}
      {children}
      {!isHomePage && !isDashboard && <Footer />}
    </>
  );
}

