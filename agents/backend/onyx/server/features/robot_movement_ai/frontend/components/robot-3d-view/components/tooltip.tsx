/**
 * Tooltip Component
 * @module robot-3d-view/components/tooltip
 */

'use client';

import { memo, useState, useRef, useEffect } from 'react';
import { getTooltipPositionClasses, type TooltipConfig } from '../utils/tooltips';

/**
 * Props for Tooltip component
 */
interface TooltipProps {
  children: React.ReactNode;
  tooltip: TooltipConfig;
}

/**
 * Tooltip Component
 * 
 * Provides accessible tooltips with proper positioning.
 * 
 * @param props - Component props
 * @returns Tooltip wrapper component
 */
export const Tooltip = memo(({ children, tooltip }: TooltipProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState(tooltip.position || 'top');
  const timeoutRef = useRef<NodeJS.Timeout>();
  const tooltipRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = () => {
    if (tooltip.disabled) return;

    if (tooltip.delay) {
      timeoutRef.current = setTimeout(() => {
        setIsVisible(true);
      }, tooltip.delay);
    } else {
      setIsVisible(true);
    }
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  // Update position based on available space
  useEffect(() => {
    if (!isVisible || !tooltipRef.current) return;

    const rect = tooltipRef.current.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Adjust position if tooltip would go off-screen
    if (rect.right > viewportWidth && position === 'right') {
      setPosition('left');
    } else if (rect.left < 0 && position === 'left') {
      setPosition('right');
    } else if (rect.bottom > viewportHeight && position === 'bottom') {
      setPosition('top');
    } else if (rect.top < 0 && position === 'top') {
      setPosition('bottom');
    }
  }, [isVisible, position]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  if (tooltip.disabled || !tooltip.content) {
    return <>{children}</>;
  }

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {isVisible && (
        <div
          ref={tooltipRef}
          className={`
            absolute z-50 px-2 py-1 text-xs text-white bg-gray-900 rounded shadow-lg
            pointer-events-none whitespace-nowrap
            ${getTooltipPositionClasses(position)}
            animate-in fade-in duration-200
          `}
          role="tooltip"
          aria-hidden={!isVisible}
        >
          {tooltip.content}
          {/* Arrow */}
          <div
            className={`
              absolute w-2 h-2 bg-gray-900 transform rotate-45
              ${position === 'top' ? 'top-full left-1/2 -translate-x-1/2 -translate-y-1/2' : ''}
              ${position === 'bottom' ? 'bottom-full left-1/2 -translate-x-1/2 translate-y-1/2' : ''}
              ${position === 'left' ? 'left-full top-1/2 -translate-y-1/2 -translate-x-1/2' : ''}
              ${position === 'right' ? 'right-full top-1/2 -translate-y-1/2 translate-x-1/2' : ''}
            `}
          />
        </div>
      )}
    </div>
  );
});

Tooltip.displayName = 'Tooltip';



