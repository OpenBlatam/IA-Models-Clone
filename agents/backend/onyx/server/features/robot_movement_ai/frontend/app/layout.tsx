import { Inter } from 'next/font/google';
import { Toaster } from 'sonner';
import ThemeProvider from '@/components/ThemeProvider';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import './globals.css';

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  preload: true,
  variable: '--font-inter',
});

export const metadata = {
  title: {
    default: 'Robot Movement AI - Control Platform',
    template: '%s | Robot Movement AI',
  },
  description: 'Plataforma IA de Movimiento Robótico - Control mediante chat',
  keywords: ['robot', 'IA', 'movimiento', 'control', 'robótica', 'automatización'],
  authors: [{ name: 'Robot Movement AI Team' }],
  creator: 'Robot Movement AI',
  publisher: 'Robot Movement AI',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'),
  openGraph: {
    type: 'website',
    locale: 'es_ES',
    url: '/',
    title: 'Robot Movement AI - Control Platform',
    description: 'Plataforma IA de Movimiento Robótico - Control mediante chat',
    siteName: 'Robot Movement AI',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Robot Movement AI - Control Platform',
    description: 'Plataforma IA de Movimiento Robótico - Control mediante chat',
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
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'default',
    title: 'Robot Movement AI',
  },
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#0062cc' },
    { media: '(prefers-color-scheme: dark)', color: '#0062cc' },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <link rel="apple-touch-icon" href="/icon-192.png" />
        <meta name="theme-color" content="#0062cc" />
      </head>
      <body className={`${inter.variable} ${inter.className} antialiased`}>
        <ErrorBoundary>
          <ThemeProvider>
            {children}
            <Toaster 
              position="top-right" 
              expand={true}
              richColors={false}
              toastOptions={{
                duration: 4000,
                classNames: {
                  toast: 'bg-white border border-gray-200 shadow-tesla-md',
                  title: 'text-tesla-black font-medium',
                  description: 'text-tesla-gray-dark',
                  success: 'bg-green-50 border-green-200',
                  error: 'bg-red-50 border-red-200',
                  warning: 'bg-yellow-50 border-yellow-200',
                  info: 'bg-blue-50 border-blue-200',
                },
              }}
            />
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}

