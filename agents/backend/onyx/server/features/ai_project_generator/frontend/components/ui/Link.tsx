'use client'

import { ReactNode } from 'react'
import NextLink from 'next/link'
import { cn } from '@/lib/utils'

interface LinkProps {
  href: string
  children: ReactNode
  className?: string
  variant?: 'default' | 'primary' | 'underline'
  external?: boolean
}

const Link = ({ href, children, className, variant = 'default', external = false }: LinkProps) => {
  const variantClasses = {
    default: 'text-gray-900 hover:text-primary-600',
    primary: 'text-primary-600 hover:text-primary-700',
    underline: 'text-primary-600 hover:text-primary-700 underline',
  }

  const baseClasses = 'transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 rounded'

  if (external) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(baseClasses, variantClasses[variant], className)}
      >
        {children}
      </a>
    )
  }

  return (
    <NextLink href={href} className={cn(baseClasses, variantClasses[variant], className)}>
      {children}
    </NextLink>
  )
}

export default Link

