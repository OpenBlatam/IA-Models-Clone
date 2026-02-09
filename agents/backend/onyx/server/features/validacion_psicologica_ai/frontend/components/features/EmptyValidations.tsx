/**
 * Empty state component for validations
 */

'use client';

import React from 'react';
import { EmptyState, Button } from '@/components/ui';
import { FileQuestion, Plus } from 'lucide-react';
import Link from 'next/link';

export const EmptyValidations: React.FC = () => {
  const handleKeyDown = (event: React.KeyboardEvent<HTMLAnchorElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      window.location.href = '/';
    }
  };

  return (
    <EmptyState
      title="No hay validaciones"
      description="Crea tu primera validación psicológica para comenzar el análisis de tus redes sociales."
      icon={<FileQuestion className="h-12 w-12 text-muted-foreground" />}
      action={
        <Link href="/" onKeyDown={handleKeyDown} tabIndex={0}>
          <Button variant="primary" aria-label="Crear nueva validación">
            <Plus className="h-4 w-4 mr-2" aria-hidden="true" />
            Crear Validación
          </Button>
        </Link>
      }
    />
  );
};




