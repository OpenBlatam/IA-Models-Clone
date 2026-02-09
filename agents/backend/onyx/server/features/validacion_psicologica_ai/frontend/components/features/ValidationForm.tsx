/**
 * Form component for creating new validations
 */

'use client';

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Button,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Select,
  Checkbox,
  PlatformButton,
} from '@/components/ui';
import { useCreateValidation } from '@/hooks/useValidations';
import type { SocialMediaPlatform } from '@/lib/types';

const validationSchema = z.object({
  platforms: z.array(z.nativeEnum(SocialMediaPlatform)).min(1, 'Selecciona al menos una plataforma'),
  include_historical_data: z.boolean().default(true),
  analysis_depth: z.enum(['basic', 'standard', 'deep']).default('standard'),
});

type ValidationFormData = z.infer<typeof validationSchema>;

const PLATFORMS: { value: SocialMediaPlatform; label: string }[] = [
  { value: SocialMediaPlatform.FACEBOOK, label: 'Facebook' },
  { value: SocialMediaPlatform.TWITTER, label: 'Twitter/X' },
  { value: SocialMediaPlatform.INSTAGRAM, label: 'Instagram' },
  { value: SocialMediaPlatform.LINKEDIN, label: 'LinkedIn' },
  { value: SocialMediaPlatform.TIKTOK, label: 'TikTok' },
  { value: SocialMediaPlatform.YOUTUBE, label: 'YouTube' },
  { value: SocialMediaPlatform.REDDIT, label: 'Reddit' },
  { value: SocialMediaPlatform.DISCORD, label: 'Discord' },
  { value: SocialMediaPlatform.TELEGRAM, label: 'Telegram' },
];

export const ValidationForm: React.FC = () => {
  const [selectedPlatforms, setSelectedPlatforms] = useState<SocialMediaPlatform[]>([]);
  const createValidation = useCreateValidation();

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
  } = useForm<ValidationFormData>({
    resolver: zodResolver(validationSchema),
    defaultValues: {
      platforms: [],
      include_historical_data: true,
      analysis_depth: 'standard',
    },
  });

  const handleTogglePlatform = (platform: SocialMediaPlatform) => {
    const newPlatforms = selectedPlatforms.includes(platform)
      ? selectedPlatforms.filter((p) => p !== platform)
      : [...selectedPlatforms, platform];
    setSelectedPlatforms(newPlatforms);
    setValue('platforms', newPlatforms, { shouldValidate: true });
  };

  const handleFormSubmit = (data: ValidationFormData) => {
    if (!data.platforms || data.platforms.length === 0) {
      return;
    }
    createValidation.mutate(data);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Nueva Validación Psicológica</CardTitle>
        <CardDescription>
          Crea una nueva validación seleccionando las redes sociales a analizar
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-3" id="platforms-label">
              Plataformas de Redes Sociales
            </label>
            <div
              className="grid grid-cols-2 md:grid-cols-3 gap-3"
              role="group"
              aria-labelledby="platforms-label"
            >
              {PLATFORMS.map((platform) => {
                const isSelected = selectedPlatforms.includes(platform.value);
                return (
                  <PlatformButton
                    key={platform.value}
                    platform={platform.label}
                    isSelected={isSelected}
                    onToggle={() => handleTogglePlatform(platform.value)}
                  />
                );
              })}
            </div>
            {errors.platforms && (
              <p className="mt-2 text-sm text-destructive" role="alert" id="platforms-error">
                {errors.platforms.message}
              </p>
            )}
          </div>

          <Checkbox
            id="historical-data"
            label="Incluir datos históricos"
            {...register('include_historical_data')}
          />

          <Select
            id="analysis-depth"
            label="Profundidad del Análisis"
            options={[
              { value: 'basic', label: 'Básico' },
              { value: 'standard', label: 'Estándar' },
              { value: 'deep', label: 'Profundo' },
            ]}
            {...register('analysis_depth')}
            error={errors.analysis_depth?.message}
          />

          <Button
            type="submit"
            variant="primary"
            className="w-full"
            isLoading={createValidation.isPending}
          >
            Crear Validación
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

