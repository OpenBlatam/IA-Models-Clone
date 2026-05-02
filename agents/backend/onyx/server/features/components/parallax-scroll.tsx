import { useEffect, useRef } from "react"
import { motion, useScroll, useTransform } from "framer-motion"

interface ParallaxScrollProps {
  children: React.ReactNode
  className?: string
  speed?: number
}

export function ParallaxScroll({
  children,
  className = "",
  speed = 0.5,
}: ParallaxScrollProps) {
  const ref = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"],
  })

  const y = useTransform(scrollYProgress, [0, 1], [0, -100 * speed])

  return (
    <div ref={ref} className={`relative overflow-hidden ${className}`}>
      <motion.div
        style={{ y }}
        className="relative"
      >
        {children}
      </motion.div>
    </div>
  )
}   