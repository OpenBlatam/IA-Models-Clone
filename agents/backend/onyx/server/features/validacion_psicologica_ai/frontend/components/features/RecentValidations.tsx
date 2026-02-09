/**
 * Recent validations component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Badge, Skeleton } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { format } from 'date-fns';
import Link from 'next/link';
import { Clock, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export const RecentValidations: React.FC = () => {
  const { data: validations, isLoading } = useValidations();

  const recentValidations = React.useMemo(() => {
    if (!validations) {
      return [];
    }
    return [...validations]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 5);
  }, [validations]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Validaciones Recientes</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!recentValidations || recentValidations.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Validaciones Recientes</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-muted-foreground py-8">
            No hay validaciones recientes
          </p>
        </CardContent>
      </Card>
    );
  }

  const getStatusVariant = (status: string): 'success' | 'destructive' | 'warning' | 'info' | 'default' => {
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
    <Card>
      <CardHeader>
        <CardTitle>Validaciones Recientes</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {recentValidations.map((validation) => (
            <Link
              key={validation.id}
              href={`/validations/${validation.id}`}
              className="block p-3 border rounded-lg hover:bg-accent transition-colors"
              aria-label={`Ver validación ${validation.id.slice(0, 8)}`}
              tabIndex={0}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm truncate">
                      #{validation.id.slice(0, 8)}
                    </span>
                    <Badge variant={getStatusVariant(validation.status)} className="text-xs">
                      {validation.status}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" aria-hidden="true" />
                    <time dateTime={validation.created_at}>
                      {format(new Date(validation.created_at), 'PPp')}
                    </time>
                  </div>
                </div>
                <ExternalLink className="h-4 w-4 text-muted-foreground flex-shrink-0 ml-2" aria-hidden="true" />
              </div>
            </Link>
          ))}
        </div>
        {validations && validations.length > 5 && (
          <div className="mt-4 pt-4 border-t">
            <Link
              href="/"
              className="text-sm text-primary hover:underline"
              aria-label="Ver todas las validaciones"
              tabIndex={0}
            >
              Ver todas las validaciones ({validations.length})
            </Link>
          </div>
        )}
      </CardContent>
    </Card>
  );
};




