'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface DataListItem {
  label: string;
  value: ReactNode;
}

interface DataListProps {
  items: DataListItem[];
  className?: string;
}

const DataList = ({ items, className }: DataListProps) => {
  return (
    <dl className={cn('space-y-3', className)}>
      {items.map((item, index) => (
        <div key={index}>
          <dt className="text-sm font-medium text-gray-700 mb-1">{item.label}</dt>
          <dd className="text-gray-900">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
};

export { DataList };

