'use client';

import { LoadingSpinner } from '@/components/ui/loading-spinner';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  children: React.ReactNode;
}

const LoadingOverlay = ({ isLoading, message = 'Cargando...', children }: LoadingOverlayProps) => {
  if (!isLoading) {
    return <>{children}</>;
  }

  return (
    <div className="relative">
      <div className="opacity-50 pointer-events-none">{children}</div>
      <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-sm z-50">
        <LoadingSpinner message={message} />
      </div>
    </div>
  );
};

export { LoadingOverlay };

