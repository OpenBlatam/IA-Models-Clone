import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from '@/lib/contexts/AuthContext';
import { ThemeProvider } from '@/lib/contexts/ThemeContext';
import { Header } from '@/components/layout/Header';
import { ErrorBoundary } from '@/lib/components/ErrorBoundary';
import { NetworkStatus } from '@/lib/components/NetworkStatus';
import { ToastNotifications } from '@/lib/components/ToastNotifications';
import { KeyboardShortcutsProvider } from '@/components/providers/KeyboardShortcutsProvider';
import { CommandPaletteProvider } from '@/components/providers/CommandPaletteProvider';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Dermatology AI - Skin Analysis & Skincare',
  description: 'AI-powered skin analysis and personalized skincare recommendations',
  keywords: 'dermatology, skincare, AI, skin analysis, recommendations',
  authors: [{ name: 'Blatam Academy' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary>
          <ThemeProvider>
            <AuthProvider>
              <KeyboardShortcutsProvider>
                <CommandPaletteProvider>
                  <Header />
                  <main>{children}</main>
                </CommandPaletteProvider>
              </KeyboardShortcutsProvider>
              <Toaster
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: '#fff',
                    color: '#333',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                  },
                  success: {
                    iconTheme: {
                      primary: '#0ea5e9',
                      secondary: '#fff',
                    },
                  },
                  error: {
                    iconTheme: {
                      primary: '#ef4444',
                      secondary: '#fff',
                    },
                  },
                }}
              />
              <NetworkStatus />
              <ToastNotifications />
              <div id="portal-root" />
            </AuthProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}

