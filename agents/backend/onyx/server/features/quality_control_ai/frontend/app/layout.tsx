import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Providers from './providers';
import { ErrorBoundary } from 'react-error-boundary';
import { Toaster } from 'sonner';
import ErrorFallback from '@/components/ErrorFallback';
import { APP_CONFIG } from '@/config/app.config';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: `${APP_CONFIG.NAME} - Inspection System`,
  description: APP_CONFIG.DESCRIPTION,
};

const RootLayout = ({
  children,
}: {
  children: React.ReactNode;
}): JSX.Element => {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <Providers>
            {children}
            <Toaster position={APP_CONFIG.TOAST_CONFIG.POSITION} richColors />
          </Providers>
        </ErrorBoundary>
      </body>
    </html>
  );
};

export default RootLayout;

