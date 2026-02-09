/**
 * Validation progress component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, ProgressBar } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { CheckCircle2, Clock, XCircle, Play } from 'lucide-react';
import { useMemo } from 'react';

export const ValidationProgress: React.FC = () => {
  const { data: validations, isLoading } = useValidations();

  const progress = useMemo(() => {
    if (!validations || validations.length === 0) {
      return {
        completed: 0,
        running: 0,
        failed: 0,
        pending: 0,
        total: 0,
      };
    }

    return {
      completed: validations.filter((v) => v.status === 'completed').length,
      running: validations.filter((v) => v.status === 'running').length,
      failed: validations.filter((v) => v.status === 'failed').length,
      pending: validations.filter((v) => v.status === 'pending').length,
      total: validations.length,
    };
  }, [validations]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="h-32 bg-muted animate-pulse rounded" />
        </CardContent>
      </Card>
    );
  }

  const completionRate = progress.total > 0
    ? Math.round((progress.completed / progress.total) * 100)
    : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Progreso de Validaciones</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <ProgressBar
          value={completionRate}
          label="Tasa de Completación"
          showValue
          variant="success"
          size="lg"
        />

        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-3 p-3 border rounded-lg">
            <CheckCircle2 className="h-5 w-5 text-green-600" aria-hidden="true" />
            <div>
              <div className="text-sm font-medium">Completadas</div>
              <div className="text-xs text-muted-foreground">{progress.completed} validaciones</div>
            </div>
          </div>

          <div className="flex items-center gap-3 p-3 border rounded-lg">
            <Play className="h-5 w-5 text-blue-600" aria-hidden="true" />
            <div>
              <div className="text-sm font-medium">En Proceso</div>
              <div className="text-xs text-muted-foreground">{progress.running} validaciones</div>
            </div>
          </div>

          <div className="flex items-center gap-3 p-3 border rounded-lg">
            <Clock className="h-5 w-5 text-gray-600" aria-hidden="true" />
            <div>
              <div className="text-sm font-medium">Pendientes</div>
              <div className="text-xs text-muted-foreground">{progress.pending} validaciones</div>
            </div>
          </div>

          <div className="flex items-center gap-3 p-3 border rounded-lg">
            <XCircle className="h-5 w-5 text-red-600" aria-hidden="true" />
            <div>
              <div className="text-sm font-medium">Fallidas</div>
              <div className="text-xs text-muted-foreground">{progress.failed} validaciones</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};



