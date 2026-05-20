'use client';

import { useState, useEffect } from 'react';

export function useFullscreen() {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleChange);
    return () => document.removeEventListener('fullscreenchange', handleChange);
  }, []);

  const enterFullscreen = async () => {
    try {
      await document.documentElement.requestFullscreen();
    } catch (error) {
      console.error('Error entering fullscreen:', error);
    }
  };

  const exitFullscreen = async () => {
    try {
      await document.exitFullscreen();
    } catch (error) {
      console.error('Error exiting fullscreen:', error);
    }
  };

  const toggleFullscreen = () => {
    if (isFullscreen) {
      exitFullscreen();
    } else {
      enterFullscreen();
    }
  };

  return { isFullscreen, enterFullscreen, exitFullscreen, toggleFullscreen };
}

