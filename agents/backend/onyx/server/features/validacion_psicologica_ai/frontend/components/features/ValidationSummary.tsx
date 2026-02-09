/**
 * Validation summary component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Progress } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { CheckCircle2, XCircle, Clock, Play } from 'lucide-react';
import { useMemo } from 'react';

export const ValidationSummary: React.FC = () => {
  const { data: validations } = useValidations();

  const summary = useMemo(() => {
    if (!validations || validations.length === 0) {
      return {
        total: 0,
        completed: 0,
        running: 0,
        failed: 0,
        pending: 0,
        completionRate: 0,
        successRate: 0,
      };
    }

    const total = validations.length;
    const completed = validations.filter((v) => v.status === 'completed').length;
    const running = validations.filter((v) => v.status === 'running').length;
    const failed = validations.filter((v) => v.status === 'failed').length;
    const pending = validations.filter((v) => v.status === 'pending').length;
    const completionRate = total > 0 ? (completed / total) * 100 : 0;
    const successRate = total > 0 ? ((total - failed) / total) * 100 : 0;

    return {
      total,
      completed,
      running,
      failed,
      pending,
      completionRate,
      successRate,
    };
  }, [validations]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Resumen de Validaciones</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Tasa de Completación</span>
            <span className="font-medium">{summary.completionRate.toFixed(1)}%</span>
          </div>
          <Progress
            value={summary.completionRate}
            variant="success"
            size="lg"
            animated
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Tasa de Éxito</span>
            <span className="font-medium">{summary.successRate.toFixed(1)}%</span>
          </div>
          <Progress
            value={summary.successRate}
            variant={summary.successRate > 80 ? 'success' : 'warning'}
            size="lg"
            animated
          />
        </div>

        <div className="grid grid-cols-2 gap-4 pt-4 border-t">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-green-600" aria-hidden="true" />
            <div>
              <div className="text-2xl font-bold">{summary.completed}</div>
              <div className="text-xs text-muted-foreground">Completadas</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Play className="h-5 w-5 text-blue-600" aria-hidden="true" />
            <div>
              <div className="text-2xl font-bold">{summary.running}</div>
              <div className="text-xs text-muted-foreground">En Proceso</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Clock className="h-5 w-5 text-gray-600" aria-hidden="true" />
            <div>
              <div className="text-2xl font-bold">{summary.pending}</div>
              <div className="text-xs text-muted-foreground">Pendientes</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <XCircle className="h-5 w-5 text-red-600" aria-hidden="true" />
            <div>
              <div className="text-2xl font-bold">{summary.failed}</div>
              <div className="text-xs text-muted-foreground">Fallidas</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};



