/**
 * Dashboard statistics component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { CheckCircle2, Clock, XCircle, TrendingUp } from 'lucide-react';
import type { ValidationStatus } from '@/lib/types';

export const DashboardStats: React.FC = () => {
  const { data: validations } = useValidations();

  if (!validations) {
    return null;
  }

  const stats = {
    total: validations.length,
    completed: validations.filter((v) => v.status === 'completed').length,
    pending: validations.filter((v) => v.status === 'pending').length,
    failed: validations.filter((v) => v.status === 'failed').length,
  };

  const completionRate = stats.total > 0 ? (stats.completed / stats.total) * 100 : 0;

  const statCards = [
    {
      title: 'Total',
      value: stats.total,
      icon: TrendingUp,
      variant: 'default' as const,
    },
    {
      title: 'Completadas',
      value: stats.completed,
      icon: CheckCircle2,
      variant: 'success' as const,
    },
    {
      title: 'Pendientes',
      value: stats.pending,
      icon: Clock,
      variant: 'warning' as const,
    },
    {
      title: 'Fallidas',
      value: stats.failed,
      icon: XCircle,
      variant: 'destructive' as const,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {statCards.map((stat) => {
        const Icon = stat.icon;
        return (
          <Card key={stat.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              {stat.title === 'Completadas' && stats.total > 0 && (
                <p className="text-xs text-muted-foreground mt-1">
                  {completionRate.toFixed(1)}% de completación
                </p>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};




