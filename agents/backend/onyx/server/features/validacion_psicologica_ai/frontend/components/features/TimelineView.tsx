/**
 * Timeline view component for validations
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { format } from 'date-fns';
import Link from 'next/link';
import { CheckCircle2, Clock, XCircle, Play, Circle } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import type { ValidationRead } from '@/lib/types';

export const TimelineView: React.FC = () => {
  const { data: validations, isLoading } = useValidations();

  if (isLoading) {
    return <div className="text-center py-12">Cargando...</div>;
  }

  if (!validations || validations.length === 0) {
    return <div className="text-center py-12 text-muted-foreground">No hay validaciones</div>;
  }

  const sortedValidations = [...validations].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  const getStatusIcon = (status: ValidationRead['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-5 w-5 text-green-600" aria-hidden="true" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-600" aria-hidden="true" />;
      case 'running':
        return <Play className="h-5 w-5 text-blue-600 animate-pulse" aria-hidden="true" />;
      default:
        return <Clock className="h-5 w-5 text-gray-600" aria-hidden="true" />;
    }
  };

  const getStatusVariant = (status: ValidationRead['status']): 'success' | 'destructive' | 'warning' | 'info' | 'default' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'destructive';
      case 'running':
        return 'info';
      case 'cancelled':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <div className="relative">
      <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-border" aria-hidden="true" />
      <div className="space-y-6">
        {sortedValidations.map((validation, index) => (
          <div key={validation.id} className="relative flex gap-4">
            <div className="relative z-10 flex-shrink-0">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-background border-2 border-border">
                {getStatusIcon(validation.status)}
              </div>
            </div>
            <div className="flex-1 min-w-0 pb-6">
              <Link
                href={`/validations/${validation.id}`}
                className="block"
                aria-label={`Ver validación ${validation.id.slice(0, 8)}`}
                tabIndex={0}
              >
                <Card className="hover:shadow-md transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">
                        Validación #{validation.id.slice(0, 8)}
                      </CardTitle>
                      <Badge variant={getStatusVariant(validation.status)}>
                        {validation.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <time dateTime={validation.created_at}>
                          {format(new Date(validation.created_at), 'PPp')}
                        </time>
                      </div>
                      {validation.completed_at && (
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <time dateTime={validation.completed_at}>
                            Completada: {format(new Date(validation.completed_at), 'PPp')}
                          </time>
                        </div>
                      )}
                      <div className="flex gap-2 flex-wrap">
                        {validation.connected_platforms.map((platform) => (
                          <Badge key={platform} variant="secondary" className="text-xs">
                            {platform}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
