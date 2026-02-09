/**
 * Platform statistics component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import { useConnections } from '@/hooks/useConnections';
import { Badge } from '@/components/ui';
import { Link2, CheckCircle2, XCircle } from 'lucide-react';

export const PlatformStats: React.FC = () => {
  const { data: connections, isLoading } = useConnections();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Plataformas Conectadas</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-muted-foreground py-8">Cargando...</p>
        </CardContent>
      </Card>
    );
  }

  if (!connections || connections.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Plataformas Conectadas</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Link2 className="h-12 w-12 text-muted-foreground mx-auto mb-4" aria-hidden="true" />
            <p className="text-muted-foreground mb-4">No hay plataformas conectadas</p>
            <a
              href="/connections"
              className="text-sm text-primary hover:underline"
              aria-label="Conectar plataformas"
              tabIndex={0}
            >
              Conectar plataformas
            </a>
          </div>
        </CardContent>
      </Card>
    );
  }

  const connectedCount = connections.filter((c) => c.status === 'connected').length;
  const totalCount = connections.length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Plataformas Conectadas</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Total</span>
            <Badge variant="default">{totalCount}</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Conectadas</span>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-600" aria-hidden="true" />
              <Badge variant="success">{connectedCount}</Badge>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Desconectadas</span>
            <div className="flex items-center gap-2">
              <XCircle className="h-4 w-4 text-red-600" aria-hidden="true" />
              <Badge variant="destructive">{totalCount - connectedCount}</Badge>
            </div>
          </div>
          <div className="pt-4 border-t">
            <a
              href="/connections"
              className="text-sm text-primary hover:underline"
              aria-label="Gestionar conexiones"
              tabIndex={0}
            >
              Gestionar conexiones →
            </a>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};




