'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ConfettiProps {
  trigger?: boolean;
  count?: number;
  colors?: string[];
}

export const Confetti = ({
  trigger = false,
  count = 50,
  colors = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
}: ConfettiProps) => {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; color: string }>>([]);

  useEffect(() => {
    if (trigger) {
      const newParticles = Array.from({ length: count }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        color: colors[Math.floor(Math.random() * colors.length)],
      }));
      setParticles(newParticles);

      setTimeout(() => setParticles([]), 3000);
    }
  }, [trigger, count, colors]);

  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      <AnimatePresence>
        {particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-2 h-2 rounded-full"
            style={{
              backgroundColor: particle.color,
              left: `${particle.x}%`,
              top: '-10px',
            }}
            animate={{
              y: (typeof window !== 'undefined' ? window.innerHeight : 1000) + 100,
              rotate: 360,
              x: particle.x + (Math.random() - 0.5) * 100,
            }}
            transition={{
              duration: Math.random() * 2 + 1,
              ease: 'easeOut',
            }}
            exit={{ opacity: 0 }}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

