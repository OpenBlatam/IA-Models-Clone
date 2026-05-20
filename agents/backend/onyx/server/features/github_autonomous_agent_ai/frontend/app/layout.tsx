import './globals.css'
import { Toaster } from './components/Toaster'

export const metadata = {
  title: 'bulk',
  description: 'una ia que no para agentes que no paran',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es">
      <body className="bg-white text-black min-h-screen">
        {children}
        <Toaster />
      </body>
    </html>
  )
}


