/**
 * Statistics chart component for dashboard
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { ValidationRead } from '@/lib/types';
import { format } from 'date-fns';

export interface StatsChartProps {
  validations: ValidationRead[];
  type?: 'line' | 'bar';
}

export const StatsChart: React.FC<StatsChartProps> = ({ validations, type = 'line' }) => {
  const chartData = React.useMemo(() => {
    const grouped = validations.reduce((acc, validation) => {
      const date = format(new Date(validation.created_at), 'yyyy-MM-dd');
      if (!acc[date]) {
        acc[date] = { date, completed: 0, failed: 0, pending: 0, running: 0 };
      }
      acc[date][validation.status] = (acc[date][validation.status] || 0) + 1;
      return acc;
    }, {} as Record<string, { date: string; completed: number; failed: number; pending: number; running: number }>);

    return Object.values(grouped).sort((a, b) => a.date.localeCompare(b.date));
  }, [validations]);

  if (chartData.length === 0) {
    return (
      <Card>
        <CardContent className="py-12">
          <p className="text-center text-muted-foreground">No hay datos para mostrar</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tendencias de Validaciones</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          {type === 'line' ? (
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="completed" stroke="#22c55e" name="Completadas" />
              <Line type="monotone" dataKey="failed" stroke="#ef4444" name="Fallidas" />
              <Line type="monotone" dataKey="pending" stroke="#6b7280" name="Pendientes" />
              <Line type="monotone" dataKey="running" stroke="#3b82f6" name="En Proceso" />
            </LineChart>
          ) : (
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="completed" fill="#22c55e" name="Completadas" />
              <Bar dataKey="failed" fill="#ef4444" name="Fallidas" />
              <Bar dataKey="pending" fill="#6b7280" name="Pendientes" />
              <Bar dataKey="running" fill="#3b82f6" name="En Proceso" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};




