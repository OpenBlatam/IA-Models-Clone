import { useState, useCallback } from 'react';
import { useIdentityOperations } from './useIdentityOperations';
import { trimFormValues } from '@/lib/utils';
import type { BuildIdentityRequest, BuildIdentityResponse } from '@/types';

interface IdentityFormValues {
  tiktokUsername: string;
  instagramUsername: string;
  youtubeChannelId: string;
}

const initialValues: IdentityFormValues = {
  tiktokUsername: '',
  instagramUsername: '',
  youtubeChannelId: '',
};

export const useIdentityForm = () => {
  const [values, setValues] = useState<IdentityFormValues>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof IdentityFormValues, string>>>({});
  const { buildIdentity, isBuilding } = useIdentityOperations();

  const handleChange = useCallback((field: keyof IdentityFormValues) => {
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
    const trimmed = trimFormValues(values);
    const newErrors: Partial<Record<keyof IdentityFormValues, string>> = {};

    const hasAtLeastOne = Object.values(trimmed).some((value) => value.length > 0);
    if (!hasAtLeastOne) {
      newErrors.tiktokUsername = 'Please provide at least one profile';
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

      const trimmed = trimFormValues(values);
      const request: BuildIdentityRequest = {};

      if (trimmed.tiktokUsername) {
        request.tiktok_username = trimmed.tiktokUsername;
      }
      if (trimmed.instagramUsername) {
        request.instagram_username = trimmed.instagramUsername;
      }
      if (trimmed.youtubeChannelId) {
        request.youtube_channel_id = trimmed.youtubeChannelId;
      }

      return await buildIdentity(request);
    },
    [values, validate, buildIdentity]
  );

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
  }, []);

  return {
    values,
    errors,
    handleChange,
    handleSubmit: handleSubmit as (e: React.FormEvent<HTMLFormElement>) => Promise<BuildIdentityResponse | undefined>,
    reset,
    isSubmitting: isBuilding,
  };
};

