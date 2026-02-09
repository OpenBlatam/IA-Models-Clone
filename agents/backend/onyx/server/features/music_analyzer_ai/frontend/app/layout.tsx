/**
 * Root layout component for the Next.js application.
 * Defines the HTML structure, metadata, and includes global providers.
 * Enhanced with better SEO, accessibility, and performance optimizations.
 */

import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { Toaster } from 'react-hot-toast';
import { ConditionalLayout } from '@/components/conditional-layout';
import { ScrollToTop } from '@/components/scroll-to-top';
import { SkipLink } from '@/components/ui';
import { appConfig } from '@/lib/config/app';
import dynamic from 'next/dynamic';

// Dynamically import API status to avoid SSR issues
const ApiStatus = dynamic(
  () =>
    import('@/components/api-status').then((mod) => ({
      default: mod.ApiStatus,
    })),
  { ssr: false }
);

// Optimize font loading
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
  preload: true,
});

/**
 * Viewport configuration for responsive design.
 */
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#1e1b4b' },
  ],
};

/**
 * Application metadata for SEO and social sharing.
 */
export const metadata: Metadata = {
  metadataBase: new URL(appConfig.url),
  title: {
    default: appConfig.name,
    template: `%s | ${appConfig.name}`,
  },
  description: appConfig.description,
  keywords: appConfig.keywords,
  authors: [{ name: appConfig.author }],
  creator: appConfig.author,
  publisher: appConfig.author,
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    type: 'website',
    locale: 'es_ES',
    url: appConfig.url,
    siteName: appConfig.name,
    title: appConfig.name,
    description: appConfig.description,
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: appConfig.name,
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: appConfig.name,
    description: appConfig.description,
    images: ['/og-image.jpg'],
    creator: '@blatamacademy',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
  manifest: '/manifest.json',
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_VERIFICATION,
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

/**
 * Root layout component.
 * Provides the base HTML structure and wraps the app with providers.
 * Optimized for SEO, accessibility, and performance.
 *
 * @param props - Component props
 * @returns Root layout component
 */
export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="es" className={inter.variable} suppressHydrationWarning>
      <head>
        {/* Preconnect to external domains for performance */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        {/* DNS prefetch for API */}
        <link rel="dns-prefetch" href={process.env.NEXT_PUBLIC_MUSIC_API_URL} />
      </head>
      <body className={inter.className} suppressHydrationWarning>
        <Providers>
          <SkipLink />
          <ApiStatus position="top-right" />
          <ConditionalLayout>
            <main id="main-content" role="main" className="min-h-screen flex flex-col">
              {children}
            </main>
          </ConditionalLayout>
          <ScrollToTop />
          <Toaster
            position={appConfig.ui.toast.position}
            toastOptions={{
              duration: appConfig.ui.toast.duration,
              style: {
                background: '#1e1b4b',
                color: '#fff',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '0.5rem',
                padding: '1rem',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#fff',
                },
                duration: appConfig.ui.toast.successDuration,
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
                duration: appConfig.ui.toast.errorDuration,
              },
            }}
            containerClassName="toast-container"
            containerStyle={{
              zIndex: 9999,
            }}
          />
        </Providers>
      </body>
    </html>
  );
}
