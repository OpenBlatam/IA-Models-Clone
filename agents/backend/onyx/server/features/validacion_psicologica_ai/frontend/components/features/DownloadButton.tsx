/**
 * Download button component for files
 */

'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui';
import { Download } from 'lucide-react';
import toast from 'react-hot-toast';

export interface DownloadButtonProps {
  url: string;
  filename: string;
  label?: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const DownloadButton: React.FC<DownloadButtonProps> = ({
  url,
  filename,
  label = 'Descargar',
  variant = 'outline',
  size = 'sm',
  className,
}) => {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (!url) {
      toast.error('URL no válida');
      return;
    }

    setIsDownloading(true);

    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
      toast.success('Descarga iniciada');
    } catch (error) {
      toast.error('Error al descargar archivo');
    } finally {
      setIsDownloading(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleDownload();
    }
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleDownload}
      onKeyDown={handleKeyDown}
      isLoading={isDownloading}
      className={className}
      aria-label={`${label}: ${filename}`}
      tabIndex={0}
    >
      <Download className="h-4 w-4 mr-2" aria-hidden="true" />
      {label}
    </Button>
  );
};

export { DownloadButton };




