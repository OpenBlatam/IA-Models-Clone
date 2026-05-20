import type { Metadata } from 'next';
import './globals.css';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import LoadingOptimizer from '@/components/LoadingOptimizer';

export const metadata: Metadata = {
  title: 'BUL - Generador de Documentos con IA',
  description: 'Crea documentos personalizados usando inteligencia artificial',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <body>
        <ErrorBoundary>
          <LoadingOptimizer />
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}

