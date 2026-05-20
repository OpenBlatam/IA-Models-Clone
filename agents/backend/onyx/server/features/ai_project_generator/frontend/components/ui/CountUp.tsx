'use client'

import { useEffect, useRef } from 'react'
import { useInView } from 'react-intersection-observer'

interface CountUpProps {
  end: number
  duration?: number
  className?: string
  prefix?: string
  suffix?: string
  decimals?: number
}

const CountUp = ({ end, duration = 2000, className, prefix = '', suffix = '', decimals = 0 }: CountUpProps) => {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.5 })
  const countRef = useRef<HTMLSpanElement>(null)
  const startTimeRef = useRef<number | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  useEffect(() => {
    if (!inView || !countRef.current) {
      return
    }

    const startTime = Date.now()
    startTimeRef.current = startTime

    const animate = () => {
      if (!countRef.current || !startTimeRef.current) {
        return
      }

      const elapsed = Date.now() - startTimeRef.current
      const progress = Math.min(elapsed / duration, 1)

      const easeOutQuart = 1 - Math.pow(1 - progress, 4)
      const current = easeOutQuart * end

      countRef.current.textContent = `${prefix}${current.toFixed(decimals)}${suffix}`

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate)
      } else {
        countRef.current.textContent = `${prefix}${end.toFixed(decimals)}${suffix}`
      }
    }

    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [inView, end, duration, prefix, suffix, decimals])

  return (
    <span ref={ref}>
      <span ref={countRef} className={className}>
        {prefix}0{suffix}
      </span>
    </span>
  )
}

export default CountUp

