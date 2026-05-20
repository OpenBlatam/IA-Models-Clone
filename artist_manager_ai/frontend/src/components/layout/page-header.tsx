'use client';

import { ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Plus } from 'lucide-react';
import Link from 'next/link';

interface PageHeaderProps {
  title: string;
  actionLabel?: string;
  actionHref?: string;
  onActionClick?: () => void;
  children?: ReactNode;
}

const PageHeader = ({ title, actionLabel, actionHref, onActionClick, children }: PageHeaderProps) => {
  const actionButton = actionLabel && (
    <Button variant="primary" onClick={onActionClick}>
      <Plus className="w-4 h-4 mr-2" />
      {actionLabel}
    </Button>
  );

  return (
    <div className="flex justify-between items-center mb-6">
      <h1 className="text-3xl font-bold text-gray-900">{title}</h1>
      <div className="flex items-center gap-4">
        {children}
        {actionHref ? (
          <Link href={actionHref}>{actionButton}</Link>
        ) : (
          actionButton
        )}
      </div>
    </div>
  );
};

export { PageHeader };

