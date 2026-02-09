'use client';

import { Button } from './button';
import { Loader2 } from 'lucide-react';
import type { ButtonProps } from './button';

interface SubmitButtonProps extends Omit<ButtonProps, 'type' | 'children'> {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
  asSubmit?: boolean;
}

export const SubmitButton = ({
  isLoading,
  loadingText = 'Cargando...',
  children,
  disabled,
  className,
  asSubmit = true,
  ...props
}: SubmitButtonProps): JSX.Element => {
  return (
    <Button
      type={asSubmit ? 'submit' : 'button'}
      disabled={isLoading || disabled}
      className={className}
      {...props}
    >
      {isLoading ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          {loadingText}
        </>
      ) : (
        children
      )}
    </Button>
  );
};

