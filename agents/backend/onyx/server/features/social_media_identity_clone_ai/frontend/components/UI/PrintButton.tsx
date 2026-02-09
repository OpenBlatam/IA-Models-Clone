import { memo, useCallback } from 'react';
import Button from './Button';
import { Printer } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PrintButtonProps {
  className?: string;
  variant?: 'primary' | 'secondary' | 'danger';
  content?: () => HTMLElement | null;
}

const PrintButton = memo(({
  className = '',
  variant = 'secondary',
  content,
}: PrintButtonProps): JSX.Element => {
  const handlePrint = useCallback(() => {
    if (content) {
      const element = content();
      if (element) {
        const printWindow = window.open('', '_blank');
        if (printWindow) {
          printWindow.document.write(`
            <html>
              <head>
                <title>Print</title>
                <style>
                  body { margin: 0; padding: 20px; }
                  @media print {
                    body { margin: 0; padding: 0; }
                  }
                </style>
              </head>
              <body>
                ${element.innerHTML}
              </body>
            </html>
          `);
          printWindow.document.close();
          printWindow.print();
        }
      }
    } else {
      window.print();
    }
  }, [content]);

  return (
    <Button
      onClick={handlePrint}
      variant={variant}
      className={cn('flex items-center gap-2', className)}
      aria-label="Print"
    >
      <Printer className="w-4 h-4" />
      Print
    </Button>
  );
});

PrintButton.displayName = 'PrintButton';

export default PrintButton;



