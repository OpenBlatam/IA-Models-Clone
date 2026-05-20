import React, { memo } from 'react';
import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hover?: boolean;
}

const paddingClasses = {
  none: '',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
};

export const Card: React.FC<CardProps> = memo(({
  children,
  className,
  padding = 'md',
  hover = false,
}) => {

  return (
    <div
      className={clsx(
        'bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700',
        paddingClasses[padding],
        hover && 'transition-shadow hover:shadow-lg',
        className
      )}
    >
      {children}
    </div>
  );
});

Card.displayName = 'Card';

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

export const CardHeader: React.FC<CardHeaderProps> = memo(({
  children,
  className,
}) => {
  return (
    <div className={clsx('mb-4', className)}>
      {children}
    </div>
  );
});

CardHeader.displayName = 'CardHeader';

interface CardTitleProps {
  children: React.ReactNode;
  className?: string;
}

export const CardTitle: React.FC<CardTitleProps> = memo(({
  children,
  className,
}) => {
  return (
    <h3 className={clsx('text-xl font-semibold text-gray-900 dark:text-white', className)}>
      {children}
    </h3>
  );
});

CardTitle.displayName = 'CardTitle';

interface CardContentProps {
  children: React.ReactNode;
  className?: string;
}

export const CardContent: React.FC<CardContentProps> = memo(({
  children,
  className,
}) => {
  return <div className={clsx('text-gray-600 dark:text-gray-300', className)}>{children}</div>;
});

CardContent.displayName = 'CardContent';

