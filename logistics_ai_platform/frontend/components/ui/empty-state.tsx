import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface EmptyStateProps {
  message: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

const EmptyState = ({ message, description, action, className }: EmptyStateProps) => {
  return (
    <Card className={cn('', className)}>
      <CardContent className="py-12 text-center">
        <p className="text-muted-foreground mb-2">{message}</p>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
        {action && <div className="mt-4">{action}</div>}
      </CardContent>
    </Card>
  );
};

export default EmptyState;




