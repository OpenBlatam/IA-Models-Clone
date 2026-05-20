'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { DecorativeDot } from './types';
import {
  DECORATIVE_DOTS_COUNT,
  LEFT_SIDE_THRESHOLD,
  MIDDLE_LEFT_THRESHOLD,
  LEFT_WIDTH_PERCENTAGE,
  MIDDLE_LEFT_START,
  MIDDLE_LEFT_END,
  RIGHT_SIDE_START,
} from './constants';

export function DecorativeDots() {
  // Generate decorative blue dots - comma-shaped/tear-drop on left, sparse grey dots elsewhere
  // Ultra-milimetric precision: exact distribution, sizes, colors, and animations
  // Memoized for performance optimization
  const blueDots = useMemo<DecorativeDot[]>(() => {
    return Array.from({ length: DECORATIVE_DOTS_COUNT }, (_, i) => {
      let left: string, top: string, opacity: number, width: string, height: string, rotation: string, color: string, hasGlow: boolean;
      const randomVal = Math.random();
      const rand = Math.random;

      if (randomVal < LEFT_SIDE_THRESHOLD) {
        // 60% on left side (0-30% width) - comma-shaped/tear-drop blue/black dots
        left = `${rand() * LEFT_WIDTH_PERCENTAGE}%`;
        opacity = 0.72 + rand() * 0.23;
        width = `${2.5 + rand() * 2}px`;
        height = `${3.5 + rand() * 2.5}px`;
        rotation = `${-25 + rand() * 50}deg`;
        color = rand() < 0.85 ? '#4285f4' : '#000000';
        hasGlow = color === '#4285f4' && rand() < 0.4;
      } else if (randomVal < MIDDLE_LEFT_THRESHOLD) {
        // 22% in middle-left (30-55% width) - smaller blue dots
        left = `${MIDDLE_LEFT_START + rand() * (MIDDLE_LEFT_END - MIDDLE_LEFT_START)}%`;
        opacity = 0.46 + rand() * 0.29;
        width = `${2.2 + rand() * 1.8}px`;
        height = `${2.8 + rand() * 2.2}px`;
        rotation = `${-12 + rand() * 24}deg`;
        color = '#4285f4';
        hasGlow = rand() < 0.25;
      } else {
        // 18% on right side (55-100% width) - sparse grey dots
        left = `${RIGHT_SIDE_START + rand() * (100 - RIGHT_SIDE_START)}%`;
        opacity = 0.13 + rand() * 0.19;
        width = `${1.4 + rand() * 1.1}px`;
        height = `${1.4 + rand() * 1.1}px`;
        rotation = '0deg';
        color = '#9aa0a6';
        hasGlow = false;
      }

      // More points in upper-left quadrant (70% in top 60%, with higher concentration in upper-left)
      const topRandom = rand();
      if (topRandom < 0.7) {
        top = `${rand() * 60}%`;
        const leftNum = parseFloat(left);
        const topNum = parseFloat(top);
        if (leftNum < LEFT_WIDTH_PERCENTAGE) {
          if (topNum < 40) {
            opacity = Math.min(opacity * 1.22, 0.95);
          } else if (topNum < 50) {
            opacity = Math.min(opacity * 1.11, 0.95);
          }
        }
      } else {
        top = `${60 + rand() * 40}%`;
        opacity *= 0.46;
      }

      return {
        id: i,
        left,
        top,
        delay: `${rand() * 15}s`,
        width,
        height,
        rotation,
        color,
        opacity,
        hasGlow
      };
    });
  }, []);

  return (
    <div className="decorative-dots" aria-hidden="true" role="presentation">
      {blueDots.map((dot) => {
        const glowRadius = parseFloat(dot.width) * 1.5;
        const glowOpacity = Math.round(dot.opacity * 30);
        const glowColor = `${dot.color}${glowOpacity.toString(16).padStart(2, '0')}`;
        
        return (
          <motion.div
            key={dot.id}
            className="decorative-dot"
            style={{
              left: dot.left,
              top: dot.top,
              width: dot.width,
              height: dot.height,
              background: dot.color,
              opacity: dot.opacity,
              borderRadius: '50%',
              boxShadow: dot.hasGlow && dot.color === '#4285f4' 
                ? `0 0 ${glowRadius}px ${glowColor}` 
                : 'none',
              willChange: 'transform, opacity',
            }}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ 
              opacity: dot.opacity,
              scale: 1,
              rotate: dot.rotation,
              x: [0, -5, -9, -5, 0],
              y: [0, -9, -15, -9, 0],
            }}
            transition={{
              duration: 12,
              delay: parseFloat(dot.delay),
              repeat: Infinity,
              ease: "easeInOut"
            }}
            aria-hidden="true"
          />
        );
      })}
    </div>
  );
}

