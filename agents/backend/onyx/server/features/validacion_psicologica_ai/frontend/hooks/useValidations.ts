/**
 * Custom hook for validations data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { validationsApi } from '@/lib/api/validations';
import type { ValidationCreate } from '@/lib/types';
import toast from 'react-hot-toast';

export const useValidations = () => {
  return useQuery({
    queryKey: ['validations'],
    queryFn: () => validationsApi.getAll(),
  });
};

export const useValidation = (id: string) => {
  return useQuery({
    queryKey: ['validation', id],
    queryFn: () => validationsApi.getById(id),
    enabled: !!id,
  });
};

export const useCreateValidation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ValidationCreate) => validationsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validations'] });
      toast.success('Validación creada exitosamente');
    },
    onError: (error: Error) => {
      toast.error(`Error al crear validación: ${error.message}`);
    },
  });
};

export const useRunValidation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => validationsApi.run(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['validation', id] });
      queryClient.invalidateQueries({ queryKey: ['validations'] });
      toast.success('Análisis iniciado');
    },
    onError: (error: Error) => {
      toast.error(`Error al ejecutar análisis: ${error.message}`);
    },
  });
};




