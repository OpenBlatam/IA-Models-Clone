import { useState, useCallback } from 'react';
import { useExtractProfileMutation } from '@/lib/modules/api';
import { Platform } from '@/types';

interface ExtractProfileFormValues {
  platform: Platform;
  username: string;
  useCache: boolean;
}

const initialValues: ExtractProfileFormValues = {
  platform: Platform.INSTAGRAM,
  username: '',
  useCache: true,
};

export const useExtractProfileForm = () => {
  const [values, setValues] = useState<ExtractProfileFormValues>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof ExtractProfileFormValues, string>>>({});
  const extractMutation = useExtractProfileMutation();

  const handleChange = useCallback((field: keyof ExtractProfileFormValues) => {
    if (field === 'useCache') {
      return (checked: boolean) => {
        setValues((prev) => ({ ...prev, useCache: checked }));
      };
    }

    if (field === 'platform') {
      return (e: React.ChangeEvent<HTMLSelectElement>) => {
        setValues((prev) => ({ ...prev, platform: e.target.value as Platform }));
      };
    }

    return (e: React.ChangeEvent<HTMLInputElement>) => {
      setValues((prev) => ({ ...prev, [field]: e.target.value }));
      if (errors[field]) {
        setErrors((prev) => {
          const newErrors = { ...prev };
          delete newErrors[field];
          return newErrors;
        });
      }
    };
  }, [errors]);

  const validate = useCallback((): boolean => {
    const newErrors: Partial<Record<keyof ExtractProfileFormValues, string>> = {};

    if (!values.username.trim()) {
      newErrors.username = 'Username is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [values]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();

      if (!validate()) {
        return;
      }

      return extractMutation.mutateAsync({
        platform: values.platform,
        username: values.username.trim(),
        use_cache: values.useCache,
      });
    },
    [values, validate, extractMutation]
  );

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
  }, []);

  return {
    values,
    errors,
    handleChange,
    handleSubmit,
    reset,
    isSubmitting: extractMutation.isLoading,
    result: extractMutation.data,
    error: extractMutation.error,
  };
};



