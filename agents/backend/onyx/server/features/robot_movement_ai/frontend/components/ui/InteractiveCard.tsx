'use client';

import { ReactNode, useState } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface InteractiveCardProps {
  children: ReactNode;
  className?: string;
  intensity?: number;
  glow?: boolean;
  tilt?: boolean;
}

export default function InteractiveCard({
  children,
  className,
  intensity = 15,
  glow = true,
  tilt = true,
}: InteractiveCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseXSpring = useSpring(x, { stiffness: 500, damping: 100 });
  const mouseYSpring = useSpring(y, { stiffness: 500, damping: 100 });

  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], [intensity, -intensity]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], [-intensity, intensity]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!tilt) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;
    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateX: tilt ? rotateX : 0,
        rotateY: tilt ? rotateY : 0,
        transformStyle: 'preserve-3d',
      }}
      className={cn('relative', className)}
    >
      <motion.div
        style={{
          transform: 'translateZ(50px)',
        }}
        className={cn(
          'relative bg-white rounded-lg border border-gray-200 shadow-sm transition-all duration-300',
          isHovered && 'shadow-tesla-lg',
          glow && isHovered && 'shadow-[0_0_30px_rgba(0,98,204,0.3)]'
        )}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}



