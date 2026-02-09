/**
 * Print button component
 */

'use client';

import React from 'react';
import { Button } from '@/components/ui';
import { Printer } from 'lucide-react';

export interface PrintButtonProps {
  title?: string;
  className?: string;
}

const PrintButton: React.FC<PrintButtonProps> = ({ title = 'Imprimir', className }) => {
  const handlePrint = () => {
    window.print();
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handlePrint();
    }
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handlePrint}
      onKeyDown={handleKeyDown}
      className={className}
      aria-label={title}
      tabIndex={0}
    >
      <Printer className="h-4 w-4 mr-2" aria-hidden="true" />
      {title}
    </Button>
  );
};

export { PrintButton };




