import { useId as useReactId } from 'react'

export const useId = (prefix?: string): string => {
  const id = useReactId()
  return prefix ? `${prefix}-${id}` : id
}

