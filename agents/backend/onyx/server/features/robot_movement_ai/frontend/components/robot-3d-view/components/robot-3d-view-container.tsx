/**
 * Robot 3D View Container Component
 * 
 * Server Component wrapper that handles layout and structure.
 * Separates client-side 3D rendering from server-side layout.
 * 
 * @module robot-3d-view/components/robot-3d-view-container
 */

import { Suspense } from 'react';
import { ErrorBoundary } from '../utils/error-boundary';
import { LoadingFallback } from './loading-fallback';
import { Robot3DViewClient } from './robot-3d-view-client';
import { robot3DViewPropsSchema } from '../schemas/validation-schemas';
import type { Robot3DViewProps } from '../schemas/validation-schemas';

/**
 * Robot 3D View Container
 * 
 * Server Component that wraps the client-side 3D view.
 * Handles error boundaries and loading states.
 * 
 * Implements:
 * - Props validation with Zod
 * - Responsive design (mobile-first)
 * - Accessibility attributes
 * - Error boundaries
 * 
 * @param props - Component props
 * @returns Robot 3D view container
 */
export function Robot3DViewContainer(props: Robot3DViewProps) {
  // Validate props at runtime
  const validatedProps = robot3DViewPropsSchema.parse(props);
  const { fullscreen = false, className = '' } = validatedProps;

  // Responsive container classes (mobile-first)
  const containerClassName = [
    'relative w-full',
    fullscreen ? 'h-screen' : 'h-[400px] md:h-[500px] lg:h-[600px]',
    'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900',
    'rounded-lg overflow-hidden',
    'border border-gray-700 shadow-2xl',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <ErrorBoundary>
      <div
        className={containerClassName}
        role="region"
        aria-label="Visualización 3D del robot"
      >
        <Suspense fallback={<LoadingFallback />}>
          <Robot3DViewClient fullscreen={fullscreen} />
        </Suspense>
      </div>
    </ErrorBoundary>
  );
}

