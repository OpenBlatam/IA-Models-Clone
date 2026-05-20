'use client';

import { ManualListItem } from './manual-list-item';
import { LoadingState } from '../ui/loading-state';
import { ErrorState } from '../ui/error-state';
import type { ManualListProps } from '@/lib/types/components';

export const ManualList = ({
  manuals,
  isLoading = false,
  error = null,
  emptyMessage = 'No se encontraron manuales',
}: ManualListProps): JSX.Element => {
  if (isLoading) {
    return <LoadingState title="Cargando manuales..." />;
  }

  if (error) {
    return <ErrorState title="Error" message="Error al cargar los manuales" />;
  }

  if (!manuals || manuals.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {manuals.map((manual) => (
        <ManualListItem key={manual.id} manual={manual} />
      ))}
    </div>
  );
};

