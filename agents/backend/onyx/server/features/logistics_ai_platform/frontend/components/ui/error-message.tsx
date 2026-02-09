import { AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ErrorMessageProps {
  message: string;
  className?: string;
  id?: string;
}

const ErrorMessage = ({ message, className, id }: ErrorMessageProps) => {
  return (
    <div
      id={id}
      className={cn('flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive', className)}
      role="alert"
      aria-live="polite"
    >
      <AlertCircle className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
      <p>{message}</p>
    </div>
  );
};

export default ErrorMessage;




