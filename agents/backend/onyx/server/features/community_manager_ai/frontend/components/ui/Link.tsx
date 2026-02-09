'use client';

import { forwardRef, AnchorHTMLAttributes } from 'react';
import { Link as NextIntlLink } from '@/i18n/routing';
import { cn } from '@/lib/utils';

interface LinkProps extends AnchorHTMLAttributes<HTMLAnchorElement> {
  href: string;
  external?: boolean;
  variant?: 'default' | 'primary' | 'muted' | 'underline';
}

const variantClasses = {
  default: 'text-gray-900 dark:text-gray-100 hover:text-primary-600 dark:hover:text-primary-400',
  primary: 'text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-500',
  muted: 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100',
  underline: 'text-primary-600 dark:text-primary-400 underline hover:no-underline',
};

export const Link = forwardRef<HTMLAnchorElement, LinkProps>(
  ({ href, external = false, variant = 'default', className, children, ...props }, ref) => {
    const linkClasses = cn(
      'transition-colors',
      variantClasses[variant],
      className
    );

    if (external) {
      return (
        <a
          ref={ref}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={linkClasses}
          {...props}
        >
          {children}
        </a>
      );
    }

    const { popover, ...linkProps } = props as any;
    return (
      <NextIntlLink ref={ref} href={href} className={linkClasses} {...linkProps}>
        {children}
      </NextIntlLink>
    );
  }
);

Link.displayName = 'Link';

