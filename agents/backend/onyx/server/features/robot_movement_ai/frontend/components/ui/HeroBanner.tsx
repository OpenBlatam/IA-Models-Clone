'use client';

import { motion } from 'framer-motion';
import { Button } from './Button';
import { cn } from '@/lib/utils/cn';
// Note: Using regular img tag instead of Next.js Image for flexibility

interface HeroBannerProps {
  title: string;
  subtitle?: string;
  description?: string;
  primaryAction?: {
    label: string;
    onClick: () => void;
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  backgroundImage?: string;
  overlay?: boolean;
  className?: string;
  fullHeight?: boolean;
}

export default function HeroBanner({
  title,
  subtitle,
  description,
  primaryAction,
  secondaryAction,
  backgroundImage,
  overlay = true,
  className,
  fullHeight = false,
}: HeroBannerProps) {
  return (
    <section
      className={cn(
        'relative w-full overflow-hidden',
        fullHeight ? 'min-h-screen' : 'min-h-[600px]',
        className
      )}
    >
      {/* Background Image */}
      {backgroundImage && (
        <div className="absolute inset-0 z-0">
          <img
            src={backgroundImage}
            alt={title}
            className="w-full h-full object-cover"
          />
          {overlay && (
            <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/20 to-black/60" />
          )}
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 container-tesla mx-auto px-4 md:px-6 lg:px-8 h-full flex items-center">
        <div className="max-w-3xl">
          {subtitle && (
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-tesla-blue text-sm md:text-base font-medium mb-4 uppercase tracking-wide"
            >
              {subtitle}
            </motion.p>
          )}
          
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-hero text-white mb-6 font-bold"
          >
            {title}
          </motion.h1>

          {description && (
            <motion.p
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-lg md:text-xl text-white/90 mb-8 leading-relaxed max-w-2xl"
            >
              {description}
            </motion.p>
          )}

          {(primaryAction || secondaryAction) && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex flex-col sm:flex-row gap-4"
            >
              {primaryAction && (
                <Button
                  variant="primary"
                  size="lg"
                  onClick={primaryAction.onClick}
                  className="bg-white text-tesla-black hover:bg-gray-100 min-h-[56px] px-8"
                >
                  {primaryAction.label}
                </Button>
              )}
              {secondaryAction && (
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={secondaryAction.onClick}
                  className="bg-transparent border-2 border-white text-white hover:bg-white/10 min-h-[56px] px-8"
                >
                  {secondaryAction.label}
                </Button>
              )}
            </motion.div>
          )}
        </div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 0.8 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-10"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-6 h-10 border-2 border-white/50 rounded-full flex items-start justify-center p-2"
        >
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-1 h-3 bg-white rounded-full"
          />
        </motion.div>
      </motion.div>
    </section>
  );
}

