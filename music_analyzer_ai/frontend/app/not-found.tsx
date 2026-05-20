/**
 * 404 Not Found page.
 * Enhanced with better UX, accessibility, and helpful navigation.
 */

import Link from 'next/link';
import { Home, ArrowLeft, Search } from 'lucide-react';
import { ROUTES } from '@/lib/constants';

/**
 * 404 Not Found page component.
 * Provides helpful navigation and clear error messaging.
 *
 * @returns Not Found page component
 */
export default function NotFound() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <div className="text-center max-w-md w-full">
        <div className="mb-8">
          <h1
            className="text-8xl font-bold text-white mb-4"
            aria-label="Error 404"
          >
            404
          </h1>
          <h2 className="text-2xl font-semibold text-white mb-2">
            Página no encontrada
          </h2>
          <p className="text-gray-300">
            Lo sentimos, la página que buscas no existe o ha sido movida.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            href={ROUTES.HOME}
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900"
            aria-label="Volver al inicio"
          >
            <Home className="w-5 h-5" aria-hidden="true" />
            Volver al inicio
          </Link>

          <Link
            href={ROUTES.MUSIC}
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900"
            aria-label="Ir a Music AI"
          >
            <Search className="w-5 h-5" aria-hidden="true" />
            Music AI
          </Link>

          <button
            onClick={() => window.history.back()}
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900"
            type="button"
            aria-label="Volver a la página anterior"
          >
            <ArrowLeft className="w-5 h-5" aria-hidden="true" />
            Volver
          </button>
        </div>

        <div className="mt-8 text-sm text-gray-400">
          <p>Si crees que esto es un error, por favor contacta al soporte.</p>
        </div>
      </div>
    </div>
  );
}
