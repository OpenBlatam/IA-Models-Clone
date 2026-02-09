import { memo } from 'react';
import { cn } from '@/lib/utils';
import Button from '@/components/UI/Button';

interface PageHeaderProps {
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary' | 'danger';
  };
  className?: string;
}

const PageHeader = memo(({ title, description, action, className = '' }: PageHeaderProps): JSX.Element => {
  return (
    <div className={cn('flex items-center justify-between mb-8', className)}>
      <div>
        <h1 className="text-3xl font-bold">{title}</h1>
        {description && <p className="text-gray-600 mt-2">{description}</p>}
      </div>
      {action && (
        <Button variant={action.variant || 'primary'} onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  );
});

PageHeader.displayName = 'PageHeader';

export default PageHeader;



