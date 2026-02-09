import { memo } from 'react';
import Card from '@/components/UI/Card';
import { cn } from '@/lib/utils';

interface FormSectionProps {
  title?: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}

const FormSection = memo(({ title, description, children, className = '' }: FormSectionProps): JSX.Element => {
  if (!title) {
    return <div className={cn('space-y-4', className)}>{children}</div>;
  }

  return (
    <Card title={title} className={className}>
      {description && <p className="text-sm text-gray-600 mb-4">{description}</p>}
      <div className="space-y-4">{children}</div>
    </Card>
  );
});

FormSection.displayName = 'FormSection';

export default FormSection;



