'use client';

import { memo } from 'react';
import type { CameraInfo as CameraInfoType } from '../types';
import { Badge } from '@/components/ui/Badge';
import LoadingSpinner from '@/components/LoadingSpinner';

interface CameraInfoProps {
  cameraInfo: CameraInfoType | null;
}

const CameraInfo = memo(({ cameraInfo }: CameraInfoProps): JSX.Element => {
  if (!cameraInfo) {
    return <LoadingSpinner size="sm" />;
  }

  return (
    <Badge variant={cameraInfo.streaming ? 'success' : 'default'}>
      {cameraInfo.streaming ? 'Streaming' : 'Idle'}
    </Badge>
  );
});

CameraInfo.displayName = 'CameraInfo';

export default CameraInfo;

