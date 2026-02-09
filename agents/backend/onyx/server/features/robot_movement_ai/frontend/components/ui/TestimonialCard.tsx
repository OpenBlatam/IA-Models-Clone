'use client';

import { motion } from 'framer-motion';
import { Star, Quote } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { Avatar } from './Avatar';

interface TestimonialCardProps {
  name: string;
  role: string;
  company?: string;
  content: string;
  rating?: number;
  avatar?: string;
  featured?: boolean;
  className?: string;
}

export default function TestimonialCard({
  name,
  role,
  company,
  content,
  rating = 5,
  avatar,
  featured = false,
  className,
}: TestimonialCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      whileHover={{ y: -4, scale: 1.02 }}
      className={cn(
        'bg-white rounded-lg border border-gray-200 p-6 md:p-8 shadow-sm transition-all',
        'hover:shadow-tesla-lg hover:border-gray-300',
        featured && 'ring-2 ring-tesla-blue/20',
        className
      )}
    >
      {/* Quote Icon */}
      <div className="mb-4">
        <Quote className="w-8 h-8 text-tesla-blue opacity-20" />
      </div>

      {/* Rating */}
      {rating > 0 && (
        <div className="flex items-center gap-1 mb-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Star
              key={i}
              className={cn(
                'w-4 h-4',
                i < rating
                  ? 'fill-yellow-400 text-yellow-400'
                  : 'fill-gray-200 text-gray-200'
              )}
            />
          ))}
        </div>
      )}

      {/* Content */}
      <p className="text-tesla-gray-dark mb-6 leading-relaxed">{content}</p>

      {/* Author */}
      <div className="flex items-center gap-4">
        <Avatar
          src={avatar}
          alt={name}
          fallback={name.charAt(0).toUpperCase()}
          size="md"
        />
        <div>
          <p className="font-semibold text-tesla-black">{name}</p>
          <p className="text-sm text-tesla-gray-dark">
            {role}
            {company && `, ${company}`}
          </p>
        </div>
      </div>
    </motion.div>
  );
}



