'use client';

import { Button } from '@/components/ui/button';
import Link from 'next/link';

interface FormActionsProps {
  submitLabel: string;
  cancelHref: string;
  cancelLabel?: string;
  isSubmitting?: boolean;
  onSubmit?: () => void;
}

const FormActions = ({
  submitLabel,
  cancelHref,
  cancelLabel = 'Cancelar',
  isSubmitting = false,
  onSubmit,
}: FormActionsProps) => {
  return (
    <div className="flex gap-4">
      <Button type="submit" variant="primary" disabled={isSubmitting} className="flex-1" onClick={onSubmit}>
        {isSubmitting ? 'Guardando...' : submitLabel}
      </Button>
      <Link href={cancelHref} className="flex-1">
        <Button type="button" variant="secondary" className="w-full">
          {cancelLabel}
        </Button>
      </Link>
    </div>
  );
};

export { FormActions };

