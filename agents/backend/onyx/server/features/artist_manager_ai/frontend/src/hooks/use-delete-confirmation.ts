import { useCallback } from 'react';
import { toast } from '@/lib/toast';

interface UseDeleteConfirmationOptions<T = void> {
  onConfirm: (id?: T) => Promise<void>;
  message?: string;
  successMessage?: string;
  errorMessage?: string;
}

export const useDeleteConfirmation = <T = void,>({
  onConfirm,
  message = '¿Estás seguro de que deseas eliminar este elemento?',
  successMessage = 'Elemento eliminado exitosamente',
  errorMessage = 'Error al eliminar el elemento',
}: UseDeleteConfirmationOptions<T>) => {
  const handleDelete = useCallback(
    async (id?: T) => {
      if (confirm(message)) {
        try {
          await onConfirm(id);
          toast.success(successMessage);
        } catch (error) {
          toast.error(error instanceof Error ? error.message : errorMessage);
        }
      }
    },
    [onConfirm, message, successMessage, errorMessage]
  );

  return handleDelete;
};

