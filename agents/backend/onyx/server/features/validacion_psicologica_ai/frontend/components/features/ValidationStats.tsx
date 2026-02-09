/**
 * Validation statistics component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, StatCard } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { CheckCircle2, XCircle, Clock, Play } from 'lucide-react';
import { useMemo } from 'react';

export const ValidationStats: React.FC = () => {
  const { data: validations, isLoading } = useValidations();

  const stats = useMemo(() => {
    if (!validations) {
      return {
        total: 0,
        completed: 0,
        running: 0,
        failed: 0,
        pending: 0,
        completionRate: 0,
      };
    }

    const total = validations.length;
    const completed = validations.filter((v) => v.status === 'completed').length;
    const running = validations.filter((v) => v.status === 'running').length;
    const failed = validations.filter((v) => v.status === 'failed').length;
    const pending = validations.filter((v) => v.status === 'pending').length;
    const completionRate = total > 0 ? Math.round((completed / total) * 100) : 0;

    return {
      total,
      completed,
      running,
      failed,
      pending,
      completionRate,
    };
  }, [validations]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="h-20 bg-muted animate-pulse rounded" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        title="Total"
        value={stats.total}
        description="Validaciones totales"
        icon={<Clock className="h-4 w-4" />}
      />
      <StatCard
        title="Completadas"
        value={stats.completed}
        description={`${stats.completionRate}% de tasa de completación`}
        icon={<CheckCircle2 className="h-4 w-4" />}
        trend={
          stats.completionRate > 50
            ? { value: stats.completionRate, isPositive: true }
            : { value: 100 - stats.completionRate, isPositive: false }
        }
      />
      <StatCard
        title="En Proceso"
        value={stats.running}
        description="Validaciones activas"
        icon={<Play className="h-4 w-4" />}
      />
      <StatCard
        title="Fallidas"
        value={stats.failed}
        description="Validaciones con error"
        icon={<XCircle className="h-4 w-4" />}
      />
    </div>
  );
};



