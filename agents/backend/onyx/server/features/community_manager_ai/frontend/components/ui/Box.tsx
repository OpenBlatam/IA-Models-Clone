'use client';

import { ReactNode, HTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

interface BoxProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  as?: 'div' | 'section' | 'article' | 'aside' | 'header' | 'footer' | 'main' | 'nav';
  padding?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  margin?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'full';
  shadow?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  border?: boolean;
  bg?: 'transparent' | 'white' | 'gray' | 'primary';
}

const paddingClasses = {
  none: '',
  xs: 'p-1',
  sm: 'p-2',
  md: 'p-4',
  lg: 'p-6',
  xl: 'p-8',
};

const marginClasses = {
  none: '',
  xs: 'm-1',
  sm: 'm-2',
  md: 'm-4',
  lg: 'm-6',
  xl: 'm-8',
};

const roundedClasses = {
  none: '',
  sm: 'rounded-sm',
  md: 'rounded-md',
  lg: 'rounded-lg',
  xl: 'rounded-xl',
  full: 'rounded-full',
};

const shadowClasses = {
  none: '',
  sm: 'shadow-sm',
  md: 'shadow-md',
  lg: 'shadow-lg',
  xl: 'shadow-xl',
};

const bgClasses = {
  transparent: '',
  white: 'bg-white dark:bg-gray-800',
  gray: 'bg-gray-50 dark:bg-gray-900',
  primary: 'bg-primary-50 dark:bg-primary-900/20',
};

export const Box = ({
  children,
  as: Component = 'div',
  padding = 'none',
  margin = 'none',
  rounded = 'none',
  shadow = 'none',
  border = false,
  bg = 'transparent',
  className,
  ...props
}: BoxProps) => {
  return (
    <Component
      className={cn(
        paddingClasses[padding],
        marginClasses[margin],
        roundedClasses[rounded],
        shadowClasses[shadow],
        border && 'border border-gray-200 dark:border-gray-700',
        bgClasses[bg],
        className
      )}
      {...props}
    >
      {children}
    </Component>
  );
};



