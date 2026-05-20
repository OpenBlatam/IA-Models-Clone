'use client';

import { IframeHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';
import { Loading } from './Loading';
import { useState } from 'react';

interface IframeProps extends IframeHTMLAttributes<HTMLIFrameElement> {
  showLoading?: boolean;
  className?: string;
}

export const Iframe = ({
  src,
  showLoading = true,
  className,
  ...props
}: IframeProps) => {
  const [isLoading, setIsLoading] = useState(true);

  return (
    <div className={cn('relative', className)}>
      {isLoading && showLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
          <Loading size="lg" />
        </div>
      )}
      <iframe
        src={src}
        onLoad={() => setIsLoading(false)}
        className={cn('w-full h-full', isLoading && 'invisible')}
        {...props}
      />
    </div>
  );
};



