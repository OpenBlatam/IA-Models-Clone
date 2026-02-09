'use client';

import { LoadingSpinner } from '@/components/ui/loading-spinner';

interface SuspenseFallbackProps {
  message?: string;
  fullScreen?: boolean;
}

const SuspenseFallback = ({ message = 'Cargando...', fullScreen = false }: SuspenseFallbackProps) => {
  return <LoadingSpinner message={message} fullScreen={fullScreen} />;
};

export { SuspenseFallback };

