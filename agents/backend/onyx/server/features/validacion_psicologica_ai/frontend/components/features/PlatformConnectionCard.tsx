/**
 * Platform connection card component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button } from '@/components/ui';
import { ConnectionStatusBadge } from './ConnectionStatusBadge';
import { format } from 'date-fns';
import type { SocialMediaConnectionResponse } from '@/lib/types';
import { Trash2, RefreshCw } from 'lucide-react';

export interface PlatformConnectionCardProps {
  connection: SocialMediaConnectionResponse;
  onDisconnect?: () => void;
  onRefresh?: () => void;
}

const PLATFORM_LABELS: Record<string, string> = {
  facebook: 'Facebook',
  twitter: 'Twitter/X',
  instagram: 'Instagram',
  linkedin: 'LinkedIn',
  tiktok: 'TikTok',
  youtube: 'YouTube',
  reddit: 'Reddit',
  discord: 'Discord',
  telegram: 'Telegram',
};

export const PlatformConnectionCard: React.FC<PlatformConnectionCardProps> = ({
  connection,
  onDisconnect,
  onRefresh,
}) => {
  const handleDisconnect = () => {
    if (onDisconnect) {
      onDisconnect();
    }
  };

  const handleRefresh = () => {
    if (onRefresh) {
      onRefresh();
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent, action: () => void) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      action();
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg capitalize">
            {PLATFORM_LABELS[connection.platform] || connection.platform}
          </CardTitle>
          <ConnectionStatusBadge status={connection.status} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {connection.connected_at && (
            <div className="text-sm text-muted-foreground">
              <time dateTime={connection.connected_at}>
                Conectado: {format(new Date(connection.connected_at), 'PPp')}
              </time>
            </div>
          )}

          <div className="flex items-center gap-2">
            {onRefresh && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                onKeyDown={(e) => handleKeyDown(e, handleRefresh)}
                aria-label="Refrescar conexión"
                tabIndex={0}
              >
                <RefreshCw className="h-4 w-4 mr-2" aria-hidden="true" />
                Refrescar
              </Button>
            )}
            {onDisconnect && (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleDisconnect}
                onKeyDown={(e) => handleKeyDown(e, handleDisconnect)}
                aria-label="Desconectar"
                tabIndex={0}
              >
                <Trash2 className="h-4 w-4 mr-2" aria-hidden="true" />
                Desconectar
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
