'use client'

import { useState, useCallback, useEffect, ReactNode } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import Button from './Button'

interface CarouselProps {
  items: ReactNode[]
  className?: string
  autoPlay?: boolean
  interval?: number
  showDots?: boolean
  showArrows?: boolean
}

const Carousel = ({
  items,
  className,
  autoPlay = false,
  interval = 5000,
  showDots = true,
  showArrows = true,
}: CarouselProps) => {
  const [currentIndex, setCurrentIndex] = useState(0)

  const handleNext = useCallback(() => {
    setCurrentIndex((prev) => (prev + 1) % items.length)
  }, [items.length])

  const handlePrevious = useCallback(() => {
    setCurrentIndex((prev) => (prev - 1 + items.length) % items.length)
  }, [items.length])

  const handleDotClick = useCallback((index: number) => {
    setCurrentIndex(index)
  }, [])

  useEffect(() => {
    if (!autoPlay) {
      return
    }

    const timer = setInterval(handleNext, interval)
    return () => clearInterval(timer)
  }, [autoPlay, interval, handleNext])

  if (items.length === 0) {
    return null
  }

  return (
    <div className={cn('relative', className)}>
      <div className="relative overflow-hidden rounded-lg">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
          >
            {items[currentIndex]}
          </motion.div>
        </AnimatePresence>
      </div>

      {showArrows && items.length > 1 && (
        <>
          <Button
            variant="secondary"
            size="sm"
            onClick={handlePrevious}
            className="absolute left-2 top-1/2 -translate-y-1/2 z-10"
            aria-label="Previous slide"
          >
            <ChevronLeft className="w-5 h-5" />
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleNext}
            className="absolute right-2 top-1/2 -translate-y-1/2 z-10"
            aria-label="Next slide"
          >
            <ChevronRight className="w-5 h-5" />
          </Button>
        </>
      )}

      {showDots && items.length > 1 && (
        <div className="flex justify-center gap-2 mt-4">
          {items.map((_, index) => (
            <button
              key={index}
              onClick={() => handleDotClick(index)}
              className={cn(
                'w-2 h-2 rounded-full transition-all',
                index === currentIndex ? 'bg-primary-600 w-6' : 'bg-gray-300'
              )}
              aria-label={`Go to slide ${index + 1}`}
              tabIndex={0}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default Carousel

