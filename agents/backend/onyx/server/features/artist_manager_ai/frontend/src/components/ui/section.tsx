'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface SectionProps {
  title?: string;
  children: ReactNode;
  className?: string;
  headerClassName?: string;
}

const Section = ({ title, children, className, headerClassName }: SectionProps) => {
  return (
    <div className={cn('mb-6', className)}>
      {title && (
        <h2 className={cn('text-xl font-semibold mb-4', headerClassName)}>{title}</h2>
      )}
      {children}
    </div>
  );
};

export { Section };

