/**
 * Hook for 3D view interactions
 * @module robot-3d-view/hooks/use-3d-interactions
 */

import { useCallback, useRef } from 'react';
import type { Position3D } from '../types';

/**
 * Options for 3D interactions
 */
interface InteractionOptions {
  /** Callback when object is clicked */
  onObjectClick?: (position: Position3D) => void;
  /** Callback when object is hovered */
  onObjectHover?: (position: Position3D) => void;
  /** Enable click interactions */
  enableClick?: boolean;
  /** Enable hover interactions */
  enableHover?: boolean;
}

/**
 * Hook for managing 3D view interactions
 * 
 * Provides handlers for click and hover events on 3D objects.
 * 
 * @param options - Interaction configuration
 * @returns Interaction handlers
 * 
 * @example
 * ```tsx
 * const { handleClick, handleHover } = use3DInteractions({
 *   onObjectClick: (pos) => console.log('Clicked:', pos),
 * });
 * ```
 */
export function use3DInteractions(options: InteractionOptions = {}) {
  const { onObjectClick, onObjectHover, enableClick = true, enableHover = true } = options;
  const hoverTimeoutRef = useRef<NodeJS.Timeout>();

  const handleClick = useCallback(
    (position: Position3D) => {
      if (!enableClick || !onObjectClick) return;
      onObjectClick(position);
    },
    [enableClick, onObjectClick]
  );

  const handleHover = useCallback(
    (position: Position3D, isEntering: boolean) => {
      if (!enableHover || !onObjectHover) return;

      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }

      if (isEntering) {
        hoverTimeoutRef.current = setTimeout(() => {
          onObjectHover(position);
        }, 100); // Debounce hover
      }
    },
    [enableHover, onObjectHover]
  );

  return {
    handleClick,
    handleHover,
  };
}



