'use client';

import { memo } from 'react';
import { Printer } from 'lucide-react';
import { usePrint } from '@/lib/hooks';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface PrintButtonProps {
  className?: string;
  variant?: 'default' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  showText?: boolean;
  onPrint?: () => void;
}

const PrintButton = memo(
  ({
    className,
    variant = 'ghost',
    size = 'icon',
    showText = false,
    onPrint,
  }: PrintButtonProps): JSX.Element => {
    const { print } = usePrint();

    const handlePrint = (): void => {
      print();
      onPrint?.();
    };

    return (
      <Button
        variant={variant}
        size={size}
        onClick={handlePrint}
        className={cn(className)}
        aria-label="Print"
      >
        <Printer className="w-4 h-4" aria-hidden="true" />
        {showText && <span className="ml-2">Print</span>}
      </Button>
    );
  }
);

PrintButton.displayName = 'PrintButton';

export default PrintButton;

