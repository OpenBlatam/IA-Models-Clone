'use client';

import { memo } from 'react';
import Link from 'next/link';
import { formatManualDate } from '@/lib/utils/format';
import { Button } from '../ui/button';
import type { ManualListItem as ManualListItemType } from '@/lib/types/api';
import type { ManualListProps } from '@/lib/types/components';

type ManualListItemProps = {
  manual: ManualListItemType;
};

export const ManualListItem = memo(({ manual }: ManualListItemProps): JSX.Element => {
  return (
    <div className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="font-semibold text-lg mb-2 line-clamp-2">
            {manual.problem_description}
          </h3>
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <span className="capitalize">{manual.category}</span>
            <span>•</span>
            <span>{formatManualDate(manual.created_at)}</span>
            {manual.model_used && (
              <>
                <span>•</span>
                <span className="truncate max-w-[150px]">
                  {manual.model_used}
                </span>
              </>
            )}
          </div>
        </div>
        <Link href={`/manual/${manual.id}`}>
          <Button variant="outline" size="sm" aria-label={`Ver manual ${manual.id}`}>
            Ver
          </Button>
        </Link>
      </div>
    </div>
  );
});

ManualListItem.displayName = 'ManualListItem';

