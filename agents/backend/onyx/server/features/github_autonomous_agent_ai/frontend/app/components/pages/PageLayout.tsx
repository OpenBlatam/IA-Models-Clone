'use client';

import { ReactNode } from 'react';
import { Header, Footer } from '../home';

interface PageLayoutProps {
  children: ReactNode;
}

export function PageLayout({ children }: PageLayoutProps) {
  return (
    <div className="min-h-screen bg-white text-black relative">
      <Header />
      <main className="relative z-10 pt-20 pb-16">
        <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
          {children}
        </div>
      </main>
      <Footer />
    </div>
  );
}

