import { memo } from 'react';
import { cn } from '@/lib/utils';

interface StatItem {
  label: string;
  value: string | number;
}

interface StatsGridProps {
  stats: StatItem[];
  columns?: 2 | 3 | 4;
  className?: string;
}

const StatsGrid = memo(({ stats, columns = 3, className = '' }: StatsGridProps): JSX.Element => {
  const gridClasses = {
    2: 'grid-cols-2',
    3: 'grid-cols-2 md:grid-cols-3',
    4: 'grid-cols-2 md:grid-cols-4',
  };

  return (
    <div className={cn('grid gap-4 pt-4 border-t', gridClasses[columns], className)}>
      {stats.map((stat, index) => (
        <div key={index}>
          <p className="text-sm text-gray-600">{stat.label}</p>
          <p className="text-2xl font-bold">{stat.value}</p>
        </div>
      ))}
    </div>
  );
});

StatsGrid.displayName = 'StatsGrid';

export default StatsGrid;



