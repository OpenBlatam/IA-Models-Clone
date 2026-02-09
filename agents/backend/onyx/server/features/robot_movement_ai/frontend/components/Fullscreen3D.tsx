'use client';

import { useState, useEffect } from 'react';
import { Maximize2, Minimize2, X } from 'lucide-react';
import Robot3DView from './Robot3DView';

export default function Fullscreen3D() {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    if (!isFullscreen) {
      const element = document.documentElement;
      if (element.requestFullscreen) {
        await element.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
      }
    }
  };

  return (
    <div className="relative w-full h-full bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={toggleFullscreen}
          className="p-3 bg-white/90 hover:bg-white border border-gray-300 rounded-md backdrop-blur-sm transition-all shadow-tesla-md min-h-[44px] min-w-[44px] flex items-center justify-center"
          title={isFullscreen ? 'Salir de pantalla completa' : 'Pantalla completa'}
          aria-label={isFullscreen ? 'Salir de pantalla completa' : 'Pantalla completa'}
        >
          {isFullscreen ? (
            <Minimize2 className="w-5 h-5 text-tesla-black" />
          ) : (
            <Maximize2 className="w-5 h-5 text-tesla-black" />
          )}
        </button>
      </div>
      <div className={isFullscreen ? 'fixed inset-0 z-50 bg-black' : 'w-full h-full'}>
        <Robot3DView />
      </div>
    </div>
  );
}

