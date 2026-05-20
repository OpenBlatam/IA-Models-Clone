/**
 * Scroll to top button component.
 * Appears when user scrolls down and provides smooth scroll to top.
 */

'use client';

import { ArrowUp } from 'lucide-react';
import { useScrollToTop } from '@/lib/hooks';
import { Button } from './ui/button';
import { cn } from '@/lib/utils';

/**
 * Scroll to top button component.
 * Provides smooth scroll to top functionality.
 *
 * @returns Scroll to top button component
 */
export function ScrollToTop() {
  const { isVisible, scrollToTop } = useScrollToTop({ threshold: 400 });

  if (!isVisible) {
    return null;
  }

  return (
    <Button
      variant="primary"
      size="md"
      onClick={scrollToTop}
      className={cn(
        'fixed bottom-8 right-8 z-50',
        'rounded-full p-3 shadow-lg',
        'transition-all duration-300',
        'hover:scale-110',
        'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2'
      )}
      aria-label="Volver arriba"
    >
      <ArrowUp className="w-5 h-5" aria-hidden="true" />
    </Button>
  );
}

