'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface FormSectionProps {
  children: React.ReactNode;
  spacing?: 2 | 4 | 6 | 8;
  className?: string;
}

const spacingMap = {
  2: 'space-y-2',
  4: 'space-y-4',
  6: 'space-y-6',
  8: 'space-y-8',
};

export const FormSection: React.FC<FormSectionProps> = memo(({
  children,
  spacing = 4,
  className,
}) => {

  return (
    <div className={clsx(spacingMap[spacing], className)}>
      {children}
    </div>
  );
});

FormSection.displayName = 'FormSection';

