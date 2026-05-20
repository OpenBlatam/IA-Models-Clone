'use client'

import { ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import Button, { ButtonProps } from './Button'

interface LoadingButtonProps extends ButtonProps {
  isLoading?: boolean
  loadingText?: string
  loadingIcon?: ReactNode
}

const LoadingButton = ({
  isLoading = false,
  loadingText,
  loadingIcon,
  children,
  disabled,
  ...props
}: LoadingButtonProps) => {
  return (
    <Button
      {...props}
      disabled={disabled || isLoading}
      leftIcon={
        isLoading ? (
          loadingIcon || <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          props.leftIcon
        )
      }
    >
      {isLoading && loadingText ? loadingText : children}
    </Button>
  )
}

export default LoadingButton

