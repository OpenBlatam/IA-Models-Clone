import { useState, useEffect, useCallback } from 'react';

export interface UseFullscreenReturn {
  isFullscreen: boolean;
  enter: () => Promise<void>;
  exit: () => Promise<void>;
  toggle: () => Promise<void>;
}

/**
 * Hook for fullscreen API
 */
export function useFullscreen(
  elementRef?: React.RefObject<HTMLElement>
): UseFullscreenReturn {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleChange);
    return () => document.removeEventListener('fullscreenchange', handleChange);
  }, []);

  const enter = useCallback(async () => {
    if (typeof window === 'undefined') return;

    try {
      const element = elementRef?.current || document.documentElement;
      if (element.requestFullscreen) {
        await element.requestFullscreen();
      } else if ((element as any).webkitRequestFullscreen) {
        await (element as any).webkitRequestFullscreen();
      } else if ((element as any).mozRequestFullScreen) {
        await (element as any).mozRequestFullScreen();
      } else if ((element as any).msRequestFullscreen) {
        await (element as any).msRequestFullscreen();
      }
    } catch (error) {
      console.error('Failed to enter fullscreen:', error);
    }
  }, [elementRef]);

  const exit = useCallback(async () => {
    if (typeof window === 'undefined') return;

    try {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
      } else if ((document as any).webkitExitFullscreen) {
        await (document as any).webkitExitFullscreen();
      } else if ((document as any).mozCancelFullScreen) {
        await (document as any).mozCancelFullScreen();
      } else if ((document as any).msExitFullscreen) {
        await (document as any).msExitFullscreen();
      }
    } catch (error) {
      console.error('Failed to exit fullscreen:', error);
    }
  }, []);

  const toggle = useCallback(async () => {
    if (isFullscreen) {
      await exit();
    } else {
      await enter();
    }
  }, [isFullscreen, enter, exit]);

  return {
    isFullscreen,
    enter,
    exit,
    toggle,
  };
}



