import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Toaster from '@/components/Toaster'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'TruthGPT Model Builder',
  description: 'Crea y despliega modelos de IA adaptados con TruthGPT',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es">
      <body className={inter.className}>
        {children}
        <Toaster />
      </body>
    </html>
  )
}

