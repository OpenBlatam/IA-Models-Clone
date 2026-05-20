'use client';

import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { clsx } from 'clsx';

interface FloatingActionButtonProps {
  icon?: React.ReactNode;
  onClick?: () => void;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  children?: React.ReactNode;
}

export const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  icon = <Plus className="h-6 w-6" />,
  onClick,
  position = 'bottom-right',
  size = 'md',
  className,
  children,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const positions = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
  };

  const sizes = {
    sm: 'h-10 w-10',
    md: 'h-14 w-14',
    lg: 'h-16 w-16',
  };

  const handleClick = () => {
    if (children) {
      setIsOpen(!isOpen);
    }
    onClick?.();
  };

  return (
    <div className={clsx('fixed z-50', positions[position])}>
      {children && isOpen && (
        <div className="mb-4 space-y-2 animate-fade-in">
          {React.Children.map(children, (child, index) => (
            <div
              key={index}
              className="animate-slide-up"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              {child}
            </div>
          ))}
        </div>
      )}
      <button
        onClick={handleClick}
        className={clsx(
          'rounded-full bg-primary-600 text-white shadow-lg hover:bg-primary-700 transition-all flex items-center justify-center',
          'hover:scale-110 active:scale-95',
          sizes[size],
          className
        )}
        aria-label="Acción flotante"
      >
        {isOpen ? <X className="h-6 w-6" /> : icon}
      </button>
    </div>
  );
};


