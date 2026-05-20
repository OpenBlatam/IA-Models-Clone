import { UseFormReset, UseFormSetValue } from 'react-hook-form';
import { showSuccessToast, showErrorToast } from './error-handler';

interface HandleSuccessOptions {
  reset?: UseFormReset<any>;
  onSuccess?: () => void;
  successMessage?: string;
  clearFiles?: () => void;
}

export const handleFormSuccess = ({
  reset,
  onSuccess,
  successMessage = 'Operación completada exitosamente',
  clearFiles,
}: HandleSuccessOptions): void => {
  showSuccessToast(successMessage);
  reset?.();
  clearFiles?.();
  onSuccess?.();
};

interface HandleErrorOptions {
  error: unknown;
  defaultMessage?: string;
}

export const handleFormError = ({
  error,
  defaultMessage = 'Ocurrió un error',
}: HandleErrorOptions): void => {
  showErrorToast(error);
};

