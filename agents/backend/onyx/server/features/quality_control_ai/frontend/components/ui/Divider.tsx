'use client';

import { memo } from 'react';
import { cn } from '@/lib/utils';
import Separator from './Separator';

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  className?: string;
  label?: string;
}

const Divider = memo(
  ({ orientation = 'horizontal', className, label }: DividerProps): JSX.Element => {
    if (label) {
      return (
        <div
          className={cn(
            'flex items-center',
            orientation === 'vertical' ? 'flex-col' : 'w-full',
            className
          )}
        >
          {orientation === 'horizontal' ? (
            <>
              <Separator className="flex-1" />
              <span className="px-4 text-sm text-gray-500">{label}</span>
              <Separator className="flex-1" />
            </>
          ) : (
            <>
              <Separator orientation="vertical" className="flex-1" />
              <span className="py-4 text-sm text-gray-500">{label}</span>
              <Separator orientation="vertical" className="flex-1" />
            </>
          )}
        </div>
      );
    }

    return <Separator orientation={orientation} className={className} />;
  }
);

Divider.displayName = 'Divider';

export default Divider;

