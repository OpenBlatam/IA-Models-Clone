'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface FeatureCardProps {
  title: string;
  description?: string;
  icon?: ReactNode;
  children?: ReactNode;
  className?: string;
  delay?: number;
  onClick?: () => void;
}

export default function FeatureCard({
  title,
  description,
  icon,
  children,
  className,
  delay = 0,
  onClick,
}: FeatureCardProps) {
  const Component = onClick ? motion.button : motion.div;

  return (
    <Component
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.5, delay }}
      whileHover={onClick ? { y: -8, scale: 1.02 } : { y: -4 }}
      onClick={onClick}
      className={cn(
        'group relative bg-white rounded-lg p-6 md:p-8 border border-gray-200 shadow-sm transition-all duration-300',
        onClick && 'cursor-pointer',
        'hover:shadow-tesla-lg hover:border-gray-300',
        className
      )}
    >
      {icon && (
        <div className="mb-4 text-tesla-blue group-hover:scale-110 transition-transform duration-300">
          {icon}
        </div>
      )}
      <h3 className="text-xl md:text-2xl font-semibold text-tesla-black mb-2 group-hover:text-tesla-blue transition-colors">
        {title}
      </h3>
      {description && (
        <p className="text-tesla-gray-dark text-sm md:text-base leading-relaxed">{description}</p>
      )}
      {children && <div className="mt-4">{children}</div>}
      <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-tesla-blue/0 to-tesla-blue/0 group-hover:from-tesla-blue/5 group-hover:to-transparent transition-all duration-300 pointer-events-none" />
    </Component>
  );
}



