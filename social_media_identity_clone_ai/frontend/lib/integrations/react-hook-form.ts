import { useForm, UseFormReturn, FieldValues, Path } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

export const createForm = <T extends FieldValues>(
  schema: z.ZodSchema<T>,
  defaultValues?: Partial<T>
): UseFormReturn<T> => {
  return useForm<T>({
    resolver: zodResolver(schema),
    defaultValues: defaultValues as T,
    mode: 'onChange',
  });
};

export const getFieldError = <T extends FieldValues>(
  form: UseFormReturn<T>,
  fieldName: Path<T>
): string | undefined => {
  return form.formState.errors[fieldName]?.message as string | undefined;
};

export const isFieldValid = <T extends FieldValues>(
  form: UseFormReturn<T>,
  fieldName: Path<T>
): boolean => {
  return !form.formState.errors[fieldName];
};



