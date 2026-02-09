import React from 'react'
import { clsx } from 'clsx'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
  header?: React.ReactNode
  footer?: React.ReactNode
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, children, header, footer, padding = 'md', ...props }, ref) => {
    const paddingClasses = {
      none: '',
      sm: 'p-4',
      md: 'p-6',
      lg: 'p-8',
    }

    return (
      <div
        ref={ref}
        className={clsx(
          'bg-white rounded-lg shadow-sm border border-gray-200',
          className
        )}
        {...props}
      >
        {header && (
          <div className="border-b border-gray-200 px-6 py-4">
            {header}
          </div>
        )}
        <div className={clsx(paddingClasses[padding])}>{children}</div>
        {footer && (
          <div className="border-t border-gray-200 px-6 py-4">{footer}</div>
        )}
      </div>
    )
  }
)

Card.displayName = 'Card'

export default Card




