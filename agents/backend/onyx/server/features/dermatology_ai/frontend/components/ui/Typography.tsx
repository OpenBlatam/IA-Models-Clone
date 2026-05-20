'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

type TypographyVariant = 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'body' | 'body2' | 'caption' | 'overline';
type TypographyColor = 'default' | 'primary' | 'secondary' | 'error' | 'success' | 'warning' | 'muted';
type TypographyAlign = 'left' | 'center' | 'right' | 'justify';

interface TypographyProps {
  variant?: TypographyVariant;
  color?: TypographyColor;
  align?: TypographyAlign;
  className?: string;
  children: React.ReactNode;
  as?: keyof JSX.IntrinsicElements;
  bold?: boolean;
  italic?: boolean;
  uppercase?: boolean;
  capitalize?: boolean;
}

const variantClasses: Record<TypographyVariant, string> = {
  h1: 'text-4xl md:text-5xl lg:text-6xl font-bold',
  h2: 'text-3xl md:text-4xl font-bold',
  h3: 'text-2xl md:text-3xl font-semibold',
  h4: 'text-xl md:text-2xl font-semibold',
  h5: 'text-lg md:text-xl font-semibold',
  h6: 'text-base md:text-lg font-semibold',
  body: 'text-base',
  body2: 'text-sm',
  caption: 'text-xs',
  overline: 'text-xs uppercase tracking-wider',
};

const colorClasses: Record<TypographyColor, string> = {
  default: 'text-gray-900 dark:text-white',
  primary: 'text-primary-600 dark:text-primary-400',
  secondary: 'text-secondary-600 dark:text-secondary-400',
  error: 'text-red-600 dark:text-red-400',
  success: 'text-green-600 dark:text-green-400',
  warning: 'text-yellow-600 dark:text-yellow-400',
  muted: 'text-gray-600 dark:text-gray-400',
};

const alignClasses: Record<TypographyAlign, string> = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
  justify: 'text-justify',
};

const getDefaultTag = (variant: TypographyVariant): keyof JSX.IntrinsicElements => {
  if (variant.startsWith('h')) {
    return variant as keyof JSX.IntrinsicElements;
  }
  return 'p';
};

export const Typography: React.FC<TypographyProps> = memo(({
  variant = 'body',
  color = 'default',
  align = 'left',
  className,
  children,
  as,
  bold = false,
  italic = false,
  uppercase = false,
  capitalize = false,
}) => {
  const Tag = as || getDefaultTag(variant);

  return (
    <Tag
      className={clsx(
        variantClasses[variant],
        colorClasses[color],
        alignClasses[align],
        bold && 'font-bold',
        italic && 'italic',
        uppercase && 'uppercase',
        capitalize && 'capitalize',
        className
      )}
    >
      {children}
    </Tag>
  );
});

Typography.displayName = 'Typography';



