'use client';

import { Card, CardContent, CardHeader, CardTitle } from './Card';
import { Skeleton } from './Skeleton';
import { EmptyState } from './EmptyState';
import { BarChart3, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChartProps {
  title: string;
  description?: string;
  data: any[];
  loading?: boolean;
  error?: string | null;
  children: React.ReactNode;
  className?: string;
}

export const Chart = ({
  title,
  description,
  data,
  loading = false,
  error = null,
  children,
  className,
}: ChartProps) => {
  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{description}</p>}
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            icon={BarChart3}
            title="Error al cargar datos"
            description={error}
          />
        </CardContent>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{description}</p>}
        </CardHeader>
        <CardContent>
          <EmptyState
            icon={TrendingUp}
            title="No hay datos"
            description="No hay datos disponibles para mostrar"
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{description}</p>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
};



