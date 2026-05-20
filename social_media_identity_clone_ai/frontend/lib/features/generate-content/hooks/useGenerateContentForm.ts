import { useState, useCallback, useMemo } from 'react';
import { useGenerateContentMutation } from '@/lib/modules/api';
import { Platform, ContentType } from '@/types';

interface GenerateContentFormValues {
  identityId: string;
  platform: Platform;
  contentType: ContentType;
  topic: string;
  style: string;
  duration: number;
  videoTitle: string;
}

const initialValues: GenerateContentFormValues = {
  identityId: '',
  platform: Platform.INSTAGRAM,
  contentType: ContentType.POST,
  topic: '',
  style: '',
  duration: 60,
  videoTitle: '',
};

export const useGenerateContentForm = () => {
  const [values, setValues] = useState<GenerateContentFormValues>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof GenerateContentFormValues, string>>>({});
  const generateMutation = useGenerateContentMutation();

  const handleChange = useCallback((field: keyof GenerateContentFormValues) => {
    if (field === 'platform' || field === 'contentType') {
      return (e: React.ChangeEvent<HTMLSelectElement>) => {
        setValues((prev) => ({ ...prev, [field]: e.target.value as Platform | ContentType }));
      };
    }

    if (field === 'duration') {
      return (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = Number(e.target.value);
        if (!isNaN(value) && value > 0) {
          setValues((prev) => ({ ...prev, duration: value }));
        }
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
    const newErrors: Partial<Record<keyof GenerateContentFormValues, string>> = {};

    if (!values.identityId.trim()) {
      newErrors.identityId = 'Identity ID is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [values]);

  const shouldShowDuration = useMemo(() => {
    return values.contentType === ContentType.VIDEO;
  }, [values.contentType]);

  const shouldShowVideoTitle = useMemo(() => {
    return values.platform === Platform.YOUTUBE;
  }, [values.platform]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();

      if (!validate()) {
        return;
      }

      const request: Parameters<typeof generateMutation.mutateAsync>[0] = {
        identity_profile_id: values.identityId.trim(),
        platform: values.platform,
        content_type: values.contentType,
      };

      const trimmedTopic = values.topic.trim();
      const trimmedStyle = values.style.trim();
      const trimmedVideoTitle = values.videoTitle.trim();

      if (trimmedTopic) {
        request.topic = trimmedTopic;
      }
      if (trimmedStyle) {
        request.style = trimmedStyle;
      }
      if (shouldShowDuration && values.duration) {
        request.duration = values.duration;
      }
      if (shouldShowVideoTitle && trimmedVideoTitle) {
        request.video_title = trimmedVideoTitle;
      }

      return generateMutation.mutateAsync(request);
    },
    [values, validate, generateMutation, shouldShowDuration, shouldShowVideoTitle]
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
    isSubmitting: generateMutation.isLoading,
    result: generateMutation.data,
    error: generateMutation.error,
    shouldShowDuration,
    shouldShowVideoTitle,
  };
};



