/**
 * Stat card component for displaying metrics
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './Card';
import { cn } from '@/lib/utils/cn';
import { TrendingUp, TrendingDown } from 'lucide-react';

export interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  icon?: React.ReactNode;
  className?: string;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  description,
  trend,
  icon,
  className,
}) => {
  return (
    <Card className={cn(className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon && <div className="text-muted-foreground" aria-hidden="true">{icon}</div>}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
        {trend && (
          <div className="flex items-center gap-1 mt-2 text-xs">
            {trend.isPositive ? (
              <TrendingUp className="h-3 w-3 text-green-600" aria-hidden="true" />
            ) : (
              <TrendingDown className="h-3 w-3 text-red-600" aria-hidden="true" />
            )}
            <span className={cn(trend.isPositive ? 'text-green-600' : 'text-red-600')}>
              {Math.abs(trend.value)}%
            </span>
            <span className="text-muted-foreground">vs período anterior</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export { StatCard };



