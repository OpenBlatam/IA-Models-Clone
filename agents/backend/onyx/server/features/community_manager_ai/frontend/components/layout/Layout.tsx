/**
 * Layout Component
 * Main application layout with sidebar and header
 * Optimized for responsive design and accessibility
 */

'use client';

import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
  className?: string;
}

/**
 * Main layout component
 * Provides the structure for the application with sidebar, header, and main content area
 */
export const Layout = ({ children, className }: LayoutProps) => {
  return (
    <ProtectedRoute>
      <div className="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
        <Sidebar />
        <div className="flex flex-1 flex-col overflow-hidden min-w-0">
          <Header />
          <main
            className={cn(
              'flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900',
              'p-4 sm:p-6',
              'focus:outline-none',
              className
            )}
            role="main"
            id="main-content"
          >
            {children}
          </main>
        </div>
      </div>
    </ProtectedRoute>
  );
};

