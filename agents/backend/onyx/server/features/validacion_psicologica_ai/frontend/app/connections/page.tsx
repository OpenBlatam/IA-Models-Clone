/**
 * Social media connections page
 */

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Trash2 } from 'lucide-react';
import {
  Button,
  LoadingSpinner,
  EmptyState,
} from '@/components/ui';
import { ConnectionForm } from '@/components/features/ConnectionForm';
import { PlatformConnectionCard } from '@/components/features/PlatformConnectionCard';
import { useConnections, useDisconnectSocialMedia } from '@/hooks/useConnections';
import type { SocialMediaPlatform } from '@/lib/types';
import { Link2 } from 'lucide-react';

const PLATFORM_LABELS: Record<SocialMediaPlatform, string> = {
  [SocialMediaPlatform.FACEBOOK]: 'Facebook',
  [SocialMediaPlatform.TWITTER]: 'Twitter/X',
  [SocialMediaPlatform.INSTAGRAM]: 'Instagram',
  [SocialMediaPlatform.LINKEDIN]: 'LinkedIn',
  [SocialMediaPlatform.TIKTOK]: 'TikTok',
  [SocialMediaPlatform.YOUTUBE]: 'YouTube',
  [SocialMediaPlatform.REDDIT]: 'Reddit',
  [SocialMediaPlatform.DISCORD]: 'Discord',
  [SocialMediaPlatform.TELEGRAM]: 'Telegram',
};


export default function ConnectionsPage() {
  const { data: connections, isLoading } = useConnections();
  const disconnectMutation = useDisconnectSocialMedia();

  const handleDisconnect = (platform: SocialMediaPlatform) => {
    if (window.confirm(`¿Estás seguro de que deseas desconectar ${PLATFORM_LABELS[platform]}?`)) {
      disconnectMutation.mutate(platform);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground"
            aria-label="Volver al inicio"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Volver</span>
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Conexiones de Redes Sociales</h1>
            <p className="text-muted-foreground">
              Gestiona tus conexiones con plataformas de redes sociales
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h2 className="text-xl font-semibold mb-4">Nueva Conexión</h2>
              <ConnectionForm />
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Conexiones Activas</h2>
              {!connections || connections.length === 0 ? (
                <Card>
                  <CardContent>
                    <EmptyState
                      title="No hay conexiones activas"
                      description="Conecta tus redes sociales para comenzar el análisis psicológico."
                      icon={<Link2 className="h-12 w-12 text-muted-foreground" />}
                    />
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {connections.map((connection) => (
                    <PlatformConnectionCard
                      key={connection.id}
                      connection={connection}
                      onDisconnect={() => handleDisconnect(connection.platform)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

