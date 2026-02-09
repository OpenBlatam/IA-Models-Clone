'use client';

import { ReactNode } from 'react';
import { StatCard } from '@/components/ui/stat-card';
import { DataGrid } from '@/components/ui/data-grid';
import { cn } from '@/lib/utils';

interface Stat {
  label: string;
  value: string | number;
  icon?: ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
}

interface StatsGridProps {
  stats: Stat[];
  columns?: 1 | 2 | 3 | 4;
  className?: string;
}

const StatsGrid = ({ stats, columns = 4, className }: StatsGridProps) => {
  return (
    <DataGrid columns={columns} gap="md" className={cn(className)}>
      {stats.map((stat, index) => (
        <StatCard
          key={index}
          title={stat.label}
          value={stat.value}
          icon={stat.icon}
          trend={stat.trend}
          variant={stat.variant}
        />
      ))}
    </DataGrid>
  );
};

export { StatsGrid };

