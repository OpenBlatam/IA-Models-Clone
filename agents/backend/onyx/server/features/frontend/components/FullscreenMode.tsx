'use client';

import { FiMaximize2, FiMinimize2 } from 'react-icons/fi';
import { useFullscreen } from '@/hooks';

interface FullscreenModeProps {
  children: React.ReactNode;
  className?: string;
}

export default function FullscreenMode({ children, className = '' }: FullscreenModeProps) {
  const { isFullscreen, toggleFullscreen } = useFullscreen();

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={toggleFullscreen}
        className="absolute top-4 right-4 z-10 btn-icon bg-white dark:bg-gray-800 shadow-lg"
        title={isFullscreen ? 'Salir de pantalla completa' : 'Pantalla completa'}
      >
        {isFullscreen ? <FiMinimize2 size={20} /> : <FiMaximize2 size={20} />}
      </button>
      {children}
    </div>
  );
}

