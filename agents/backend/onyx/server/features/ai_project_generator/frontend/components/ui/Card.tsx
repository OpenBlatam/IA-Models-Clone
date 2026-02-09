'use client'

import clsx from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
  hover?: boolean
}

const Card = ({ children, className, padding = 'md', hover = false }: CardProps) => {
  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  }

  return (
    <div
      className={clsx(
        'bg-white rounded-lg shadow-md',
        paddingClasses[padding],
        hover && 'hover:shadow-lg transition-shadow',
        className
      )}
    >
      {children}
    </div>
  )
}

export default Card

