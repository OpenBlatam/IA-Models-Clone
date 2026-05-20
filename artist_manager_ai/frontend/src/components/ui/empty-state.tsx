'use client';

import { LucideIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  actionLabel?: string;
  actionHref?: string;
  onActionClick?: () => void;
}

const EmptyState = ({
  icon: Icon,
  title,
  description,
  actionLabel,
  actionHref,
  onActionClick,
}: EmptyStateProps) => {
  return (
    <Card>
      <CardContent className="text-center py-12">
        <Icon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
        {description && <p className="text-gray-500 mb-4">{description}</p>}
        {actionLabel && (
          <div className="mt-6">
            {actionHref ? (
              <Button variant="primary" asChild>
                <a href={actionHref}>{actionLabel}</a>
              </Button>
            ) : (
              <Button variant="primary" onClick={onActionClick}>
                {actionLabel}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export { EmptyState };

