/**
 * Platform selection button component with improved accessibility
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';
import { cva, type VariantProps } from 'class-variance-authority';
import { Check } from 'lucide-react';

const platformButtonVariants = cva(
  'relative flex items-center justify-center p-4 rounded-md border-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
  {
    variants: {
      selected: {
        true: 'border-primary bg-primary/10',
        false: 'border-input hover:border-primary/50',
      },
    },
    defaultVariants: {
      selected: false,
    },
  }
);

export interface PlatformButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'>,
    VariantProps<typeof platformButtonVariants> {
  platform: string;
  isSelected: boolean;
  onToggle: () => void;
}

const PlatformButton = React.forwardRef<HTMLButtonElement, PlatformButtonProps>(
  ({ className, platform, isSelected, onToggle, ...props }, ref) => {
    const handleClick = () => {
      onToggle();
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onToggle();
      }
    };

    return (
      <button
        ref={ref}
        type="button"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        className={cn(platformButtonVariants({ selected: isSelected }), className)}
        aria-pressed={isSelected}
        aria-label={`${isSelected ? 'Deseleccionar' : 'Seleccionar'} ${platform}`}
        tabIndex={0}
        {...props}
      >
        {isSelected && (
          <Check
            className="absolute top-2 right-2 h-4 w-4 text-primary"
            aria-hidden="true"
          />
        )}
        <span className="font-medium">{platform}</span>
      </button>
    );
  }
);

PlatformButton.displayName = 'PlatformButton';

export { PlatformButton };




