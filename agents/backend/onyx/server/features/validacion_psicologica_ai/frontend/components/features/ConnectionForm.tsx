/**
 * Form component for connecting social media platforms
 */

'use client';

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
  Input,
  Select,
} from '@/components/ui';
import { useConnectSocialMedia } from '@/hooks/useConnections';
import type { SocialMediaPlatform } from '@/lib/types';

const connectionSchema = z.object({
  platform: z.nativeEnum(SocialMediaPlatform),
  access_token: z.string().min(1, 'El token de acceso es requerido'),
  refresh_token: z.string().optional(),
  expires_in: z.number().optional(),
});

type ConnectionFormData = z.infer<typeof connectionSchema>;

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

export const ConnectionForm: React.FC = () => {
  const connectMutation = useConnectSocialMedia();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<ConnectionFormData>({
    resolver: zodResolver(connectionSchema),
    defaultValues: {
      expires_in: 3600,
    },
  });

  const handleFormSubmit = (data: ConnectionFormData) => {
    connectMutation.mutate(data, {
      onSuccess: () => {
        reset();
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Conectar Red Social</CardTitle>
        <CardDescription>
          Conecta una de tus redes sociales para comenzar el análisis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-4">
          <Select
            id="platform"
            label="Plataforma"
            options={PLATFORMS.map((p) => ({ value: p.value, label: p.label }))}
            {...register('platform')}
            error={errors.platform?.message}
          />

          <Input
            id="access_token"
            label="Token de Acceso"
            type="password"
            {...register('access_token')}
            error={errors.access_token?.message}
            aria-describedby="access-token-help"
          />
          <p id="access-token-help" className="text-xs text-muted-foreground">
            Ingresa el token de acceso de la API de la plataforma
          </p>

          <Input
            id="refresh_token"
            label="Token de Refresco (Opcional)"
            type="password"
            {...register('refresh_token')}
            error={errors.refresh_token?.message}
          />

          <Input
            id="expires_in"
            label="Tiempo de Expiración (segundos)"
            type="number"
            {...register('expires_in', { valueAsNumber: true })}
            error={errors.expires_in?.message}
          />

          <Button
            type="submit"
            variant="primary"
            className="w-full"
            isLoading={connectMutation.isPending}
          >
            Conectar
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};




