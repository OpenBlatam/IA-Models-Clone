/**
 * Quick stats cards component
 */

'use client';

import React from 'react';
import { StatCard } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { CheckCircle2, XCircle, Clock, Play, TrendingUp } from 'lucide-react';
import { useMemo } from 'react';

export const QuickStatsCards: React.FC = () => {
  const { data: validations, isLoading } = useValidations();

  const stats = useMemo(() => {
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
    const completionRate = total > 0 ? Math.round((completed / total) * 100) : 0;
    const successRate = total > 0 ? Math.round(((total - failed) / total) * 100) : 0;

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

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
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
        description={`${stats.completionRate}% completadas`}
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
        title="Tasa de Éxito"
        value={`${stats.successRate}%`}
        description={`${stats.total - stats.failed} de ${stats.total} exitosas`}
        icon={<TrendingUp className="h-4 w-4" />}
        trend={
          stats.successRate > 80
            ? { value: stats.successRate, isPositive: true }
            : { value: 100 - stats.successRate, isPositive: false }
        }
      />
    </div>
  );
};



