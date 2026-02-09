import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Toaster } from 'react-hot-toast'
import Providers from './providers'
import ErrorBoundary from '@/components/ui/ErrorBoundary'
import GlobalCommandPalette from '@/components/features/GlobalCommandPalette'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Research Paper Code Improver',
  description: 'Improve your code using knowledge from research papers',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary>
          <Providers>
            {children}
            <Toaster position="top-right" />
            <GlobalCommandPalette />
          </Providers>
        </ErrorBoundary>
      </body>
    </html>
  )
}

