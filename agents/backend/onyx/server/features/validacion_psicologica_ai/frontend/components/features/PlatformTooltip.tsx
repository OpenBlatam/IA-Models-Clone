/**
 * Platform tooltip component
 */

'use client';

import React from 'react';
import { Tooltip } from '@/components/ui';
import type { SocialMediaPlatform } from '@/lib/types';

export interface PlatformTooltipProps {
  platform: SocialMediaPlatform;
  children: React.ReactElement;
}

const PLATFORM_INFO: Record<SocialMediaPlatform, { name: string; description: string }> = {
  [SocialMediaPlatform.FACEBOOK]: {
    name: 'Facebook',
    description: 'Red social para conectar con amigos y familia',
  },
  [SocialMediaPlatform.TWITTER]: {
    name: 'Twitter/X',
    description: 'Plataforma de microblogging y noticias',
  },
  [SocialMediaPlatform.INSTAGRAM]: {
    name: 'Instagram',
    description: 'Red social para compartir fotos y videos',
  },
  [SocialMediaPlatform.LINKEDIN]: {
    name: 'LinkedIn',
    description: 'Red profesional para networking',
  },
  [SocialMediaPlatform.TIKTOK]: {
    name: 'TikTok',
    description: 'Plataforma de videos cortos',
  },
  [SocialMediaPlatform.YOUTUBE]: {
    name: 'YouTube',
    description: 'Plataforma de videos y contenido',
  },
  [SocialMediaPlatform.REDDIT]: {
    name: 'Reddit',
    description: 'Comunidades y discusiones',
  },
  [SocialMediaPlatform.DISCORD]: {
    name: 'Discord',
    description: 'Plataforma de chat y comunicación',
  },
  [SocialMediaPlatform.TELEGRAM]: {
    name: 'Telegram',
    description: 'Mensajería instantánea',
  },
};

export const PlatformTooltip: React.FC<PlatformTooltipProps> = ({ platform, children }) => {
  const info = PLATFORM_INFO[platform] || { name: platform, description: 'Plataforma de redes sociales' };

  return (
    <Tooltip
      content={
        <div>
          <div className="font-medium">{info.name}</div>
          <div className="text-xs text-muted-foreground mt-1">{info.description}</div>
        </div>
      }
      position="top"
    >
      {children}
    </Tooltip>
  );
};



