/**
 * Validation metrics component with detailed statistics
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Progress } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { format } from 'date-fns';

export const ValidationMetrics: React.FC = () => {
  const { data: validations } = useValidations();

  if (!validations || validations.length === 0) {
    return null;
  }

  const metrics = React.useMemo(() => {
    const total = validations.length;
    const completed = validations.filter((v) => v.status === 'completed').length;
    const failed = validations.filter((v) => v.status === 'failed').length;
    const pending = validations.filter((v) => v.status === 'pending').length;
    const running = validations.filter((v) => v.status === 'running').length;

    const completionRate = total > 0 ? (completed / total) * 100 : 0;
    const failureRate = total > 0 ? (failed / total) * 100 : 0;

    const avgTimeToComplete = React.useMemo(() => {
      const completedValidations = validations.filter(
        (v) => v.status === 'completed' && v.completed_at
      );
      if (completedValidations.length === 0) {
        return null;
      }

      const totalTime = completedValidations.reduce((acc, v) => {
        const created = new Date(v.created_at).getTime();
        const completed = new Date(v.completed_at!).getTime();
        return acc + (completed - created);
      }, 0);

      const avgMs = totalTime / completedValidations.length;
      return Math.round(avgMs / (1000 * 60)); // minutos
    }, [validations]);

    const recentValidations = [...validations]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 5);

    return {
      total,
      completed,
      failed,
      pending,
      running,
      completionRate,
      failureRate,
      avgTimeToComplete,
      recentValidations,
    };
  }, [validations]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Tasa de Completación</CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={metrics.completionRate} label="Completadas" showValue />
          <p className="text-sm text-muted-foreground mt-2">
            {metrics.completed} de {metrics.total} validaciones completadas
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Tasa de Fallos</CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={metrics.failureRate} label="Fallidas" showValue />
          <p className="text-sm text-muted-foreground mt-2">
            {metrics.failed} de {metrics.total} validaciones fallidas
          </p>
        </CardContent>
      </Card>

      {metrics.avgTimeToComplete !== null && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Tiempo Promedio</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{metrics.avgTimeToComplete} min</div>
            <p className="text-sm text-muted-foreground mt-2">
              Tiempo promedio para completar una validación
            </p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Distribución de Estados</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Completadas</span>
                <span>{metrics.completed}</span>
              </div>
              <Progress value={(metrics.completed / metrics.total) * 100} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Pendientes</span>
                <span>{metrics.pending}</span>
              </div>
              <Progress value={(metrics.pending / metrics.total) * 100} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>En Proceso</span>
                <span>{metrics.running}</span>
              </div>
              <Progress value={(metrics.running / metrics.total) * 100} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Fallidas</span>
                <span>{metrics.failed}</span>
              </div>
              <Progress value={(metrics.failed / metrics.total) * 100} />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};




