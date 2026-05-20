import * as React from 'react';
import { cn } from '@/lib/utils';
import { Button } from './button';

interface DropdownMenuProps {
  children: React.ReactNode;
  trigger: React.ReactNode;
  align?: 'left' | 'right' | 'center';
}

const DropdownMenu = ({ children, trigger, align = 'left' }: DropdownMenuProps) => {
  const [open, setOpen] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false);
    }
  };

  const alignClasses = {
    left: 'left-0',
    right: 'right-0',
    center: 'left-1/2 -translate-x-1/2',
  };

  return (
    <div className="relative" ref={menuRef} onKeyDown={handleKeyDown}>
      <div onClick={() => setOpen(!open)} role="button" tabIndex={0} aria-expanded={open}>
        {trigger}
      </div>
      {open && (
        <div
          className={cn(
            'absolute z-50 mt-2 min-w-[8rem] rounded-md border bg-popover p-1 text-popover-foreground shadow-md',
            alignClasses[align]
          )}
          role="menu"
        >
          {children}
        </div>
      )}
    </div>
  );
};

const DropdownMenuItem = ({
  className,
  children,
  onClick,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => {
  const handleClick = (e: React.MouseEvent) => {
    onClick?.(e);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick?.(e as any);
    }
  };

  return (
    <div
      className={cn(
        'relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground',
        className
      )}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="menuitem"
      tabIndex={0}
      {...props}
    >
      {children}
    </div>
  );
};

export { DropdownMenu, DropdownMenuItem };
