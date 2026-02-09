'use client';

import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface Feature {
  icon: LucideIcon;
  title: string;
  description: string;
  highlight?: boolean;
}

interface FeatureShowcaseProps {
  features: Feature[];
  columns?: 2 | 3 | 4;
  className?: string;
  variant?: 'default' | 'cards' | 'minimal';
}

export default function FeatureShowcase({
  features,
  columns = 3,
  className,
  variant = 'default',
}: FeatureShowcaseProps) {
  const gridCols = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };

  if (variant === 'minimal') {
    return (
      <div className={cn('space-y-12', className)}>
        {features.map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className={cn(
              'flex flex-col md:flex-row items-start gap-6',
              index % 2 === 1 && 'md:flex-row-reverse'
            )}
          >
            <div className="flex-shrink-0">
              <div className="w-16 h-16 bg-tesla-blue/10 rounded-lg flex items-center justify-center">
                <feature.icon className="w-8 h-8 text-tesla-blue" />
              </div>
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-semibold text-tesla-black mb-2">
                {feature.title}
              </h3>
              <p className="text-tesla-gray-dark leading-relaxed">
                {feature.description}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    );
  }

  if (variant === 'cards') {
    return (
      <div className={cn('grid gap-6', gridCols[columns], className)}>
        {features.map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: index * 0.1 }}
            whileHover={{ y: -4, scale: 1.02 }}
            className={cn(
              'bg-white rounded-lg border border-gray-200 p-6 shadow-sm transition-all',
              'hover:shadow-tesla-lg hover:border-gray-300',
              feature.highlight && 'ring-2 ring-tesla-blue/20'
            )}
          >
            <div className="w-12 h-12 bg-tesla-blue/10 rounded-lg flex items-center justify-center mb-4">
              <feature.icon className="w-6 h-6 text-tesla-blue" />
            </div>
            <h3 className="text-lg font-semibold text-tesla-black mb-2">
              {feature.title}
            </h3>
            <p className="text-tesla-gray-dark text-sm leading-relaxed">
              {feature.description}
            </p>
          </motion.div>
        ))}
      </div>
    );
  }

  return (
    <div className={cn('grid gap-8', gridCols[columns], className)}>
      {features.map((feature, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4, delay: index * 0.1 }}
          className="text-center"
        >
          <div className="w-16 h-16 bg-tesla-blue/10 rounded-full flex items-center justify-center mx-auto mb-4">
            <feature.icon className="w-8 h-8 text-tesla-blue" />
          </div>
          <h3 className="text-xl font-semibold text-tesla-black mb-3">
            {feature.title}
          </h3>
          <p className="text-tesla-gray-dark leading-relaxed">
            {feature.description}
          </p>
        </motion.div>
      ))}
    </div>
  );
}



