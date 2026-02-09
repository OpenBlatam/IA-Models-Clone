/**
 * Loading Fallback Component
 * @module robot-3d-view/components/loading-fallback
 */

import { memo } from 'react';

/**
 * Loading fallback component for Suspense boundaries
 * 
 * Implements accessibility best practices with ARIA labels and semantic HTML.
 * 
 * @returns Loading fallback UI
 */
export const LoadingFallback = memo(() => {
  return (
    <div
      className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900"
      role="status"
      aria-live="polite"
      aria-label="Cargando vista 3D"
    >
      <div className="text-center">
        <div
          className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent text-white motion-reduce:animate-[spin_1.5s_linear_infinite]"
          aria-hidden="true"
        />
        <p className="mt-4 text-white text-lg font-medium">
          Cargando vista 3D...
        </p>
        <span className="sr-only">Cargando visualización 3D del robot</span>
      </div>
    </div>
  );
});

LoadingFallback.displayName = 'LoadingFallback';

