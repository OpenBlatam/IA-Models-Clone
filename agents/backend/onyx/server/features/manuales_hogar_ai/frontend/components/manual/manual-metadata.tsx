'use client';

import { formatManualDate } from '@/lib/utils/format';
import type { ManualMetadataProps } from '@/lib/types/components';

export const ManualMetadata = ({ manual }: ManualMetadataProps): JSX.Element => {
  return (
    <div className="flex items-center space-x-4 mt-2">
      <span className="capitalize">{manual.category}</span>
      <span>•</span>
      <span>{formatManualDate(manual.created_at)}</span>
      {manual.model_used && (
        <>
          <span>•</span>
          <span className="truncate max-w-[150px]" title={manual.model_used}>
            {manual.model_used}
          </span>
        </>
      )}
    </div>
  );
};

