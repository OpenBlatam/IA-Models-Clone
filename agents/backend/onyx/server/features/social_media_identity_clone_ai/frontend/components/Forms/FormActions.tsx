import { memo } from 'react';
import Button from '@/components/UI/Button';
import { cn } from '@/lib/utils';

interface FormActionsProps {
  onSubmit: () => void;
  onCancel?: () => void;
  submitLabel?: string;
  cancelLabel?: string;
  isLoading?: boolean;
  disabled?: boolean;
  className?: string;
}

const FormActions = memo(({
  onSubmit,
  onCancel,
  submitLabel = 'Submit',
  cancelLabel = 'Cancel',
  isLoading = false,
  disabled = false,
  className = '',
}: FormActionsProps): JSX.Element => {
  return (
    <div className={cn('flex gap-4', className)}>
      {onCancel && (
        <Button type="button" variant="secondary" onClick={onCancel} disabled={isLoading || disabled}>
          {cancelLabel}
        </Button>
      )}
      <Button type="submit" onClick={onSubmit} isLoading={isLoading} disabled={disabled} className="flex-1">
        {submitLabel}
      </Button>
    </div>
  );
});

FormActions.displayName = 'FormActions';

export default FormActions;



