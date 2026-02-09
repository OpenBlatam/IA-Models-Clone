'use client';

import { memo, type ReactNode } from 'react';
import Header from '@/components/Header';

interface LayoutProps {
  children: ReactNode;
}

const Layout = memo(({ children }: LayoutProps): JSX.Element => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-6" role="main">
        {children}
      </main>
    </div>
  );
});

Layout.displayName = 'Layout';

export default Layout;
