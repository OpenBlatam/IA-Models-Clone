import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import Navbar from '@/components/layout/navbar-enhanced';
import { OnlineIndicator } from '@/components/ui/online-indicator';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Artist Manager AI',
  description: 'Sistema de IA para gestión integral de artistas',
};

const RootLayout = ({ children }: { children: React.ReactNode }) => {
  return (
    <html lang="es">
      <body className={inter.className}>
        <Providers>
          <OnlineIndicator />
          <Navbar />
          {children}
        </Providers>
      </body>
    </html>
  );
};

export default RootLayout;

