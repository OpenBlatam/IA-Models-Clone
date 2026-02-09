/**
 * Dashboard widgets component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { TrendingUp, TrendingDown, Activity, Clock } from 'lucide-react';
import { useMemo } from 'react';

export const DashboardWidgets: React.FC = () => {
  const { data: validations } = useValidations();

  const metrics = useMemo(() => {
    if (!validations || validations.length === 0) {
      return {
        total: 0,
        completed: 0,
        running: 0,
        failed: 0,
        completionRate: 0,
        averageTime: 0,
        trend: 0,
      };
    }

    const total = validations.length;
    const completed = validations.filter((v) => v.status === 'completed').length;
    const running = validations.filter((v) => v.status === 'running').length;
    const failed = validations.filter((v) => v.status === 'failed').length;
    const completionRate = total > 0 ? (completed / total) * 100 : 0;

    const completedWithTime = validations.filter(
      (v) => v.status === 'completed' && v.completed_at
    );
    const averageTime =
      completedWithTime.length > 0
        ? completedWithTime.reduce((acc, v) => {
            const time =
              new Date(v.completed_at!).getTime() -
              new Date(v.created_at).getTime();
            return acc + time;
          }, 0) / completedWithTime.length
        : 0;

    return {
      total,
      completed,
      running,
      failed,
      completionRate: Math.round(completionRate),
      averageTime: Math.round(averageTime / (1000 * 60)), // en minutos
      trend: completionRate > 50 ? 1 : -1,
    };
  }, [validations]);

  const formatTime = (minutes: number): string => {
    if (minutes < 60) {
      return `${minutes} min`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}min`;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Validaciones</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.total}</div>
          <p className="text-xs text-muted-foreground">Validaciones creadas</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Tasa de Completación</CardTitle>
          {metrics.trend > 0 ? (
            <TrendingUp className="h-4 w-4 text-green-600" aria-hidden="true" />
          ) : (
            <TrendingDown className="h-4 w-4 text-red-600" aria-hidden="true" />
          )}
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.completionRate}%</div>
          <p className="text-xs text-muted-foreground">
            {metrics.completed} de {metrics.total} completadas
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">En Proceso</CardTitle>
          <Clock className="h-4 w-4 text-blue-600" aria-hidden="true" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.running}</div>
          <p className="text-xs text-muted-foreground">Validaciones activas</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Tiempo Promedio</CardTitle>
          <Clock className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{formatTime(metrics.averageTime)}</div>
          <p className="text-xs text-muted-foreground">Tiempo de procesamiento</p>
        </CardContent>
      </Card>
    </div>
  );
};



