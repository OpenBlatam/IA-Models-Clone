import React from 'react'
import { clsx } from 'clsx'

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular'
  width?: string | number
  height?: string | number
  animation?: 'pulse' | 'wave' | 'none'
}

const Skeleton: React.FC<SkeletonProps> = ({
  className,
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse',
  style,
  ...props
}) => {
  const baseStyles = 'bg-gray-200 rounded'
  
  const variants = {
    text: 'h-4',
    circular: 'rounded-full',
    rectangular: 'rounded',
  }

  const animations = {
    pulse: 'animate-pulse',
    wave: 'animate-pulse',
    none: '',
  }

  const customStyle = {
    width: width ? (typeof width === 'number' ? `${width}px` : width) : undefined,
    height: height ? (typeof height === 'number' ? `${height}px` : height) : undefined,
    ...style,
  }

  return (
    <div
      className={clsx(
        baseStyles,
        variants[variant],
        animations[animation],
        className
      )}
      style={customStyle}
      {...props}
    />
  )
}

export default Skeleton




