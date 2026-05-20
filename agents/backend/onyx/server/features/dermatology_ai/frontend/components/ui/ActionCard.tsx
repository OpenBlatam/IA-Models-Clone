'use client';

import React, { memo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './Card';
import { Button } from './Button';
import { clsx } from 'clsx';

interface ActionCardProps {
  title: string;
  children: React.ReactNode;
  action?: {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
    isLoading?: boolean;
    disabled?: boolean;
  };
  className?: string;
  headerClassName?: string;
  contentClassName?: string;
}

export const ActionCard: React.FC<ActionCardProps> = memo(({
  title,
  children,
  action,
  className,
  headerClassName,
  contentClassName,
}) => {
  return (
    <Card className={className}>
      <CardHeader className={headerClassName}>
        <div className="flex items-center justify-between">
          <CardTitle>{title}</CardTitle>
          {action && (
            <Button
              onClick={action.onClick}
              variant={action.variant || 'primary'}
              isLoading={action.isLoading}
              disabled={action.disabled}
              size="sm"
            >
              {action.label}
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className={contentClassName}>{children}</CardContent>
    </Card>
  );
});

ActionCard.displayName = 'ActionCard';



