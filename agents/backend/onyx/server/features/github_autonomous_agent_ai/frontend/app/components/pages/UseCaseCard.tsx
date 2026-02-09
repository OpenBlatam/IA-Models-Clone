'use client';

import { motion } from 'framer-motion';

interface UseCaseCardProps {
  id: string;
  title: string;
  description: string;
  index?: number;
}

export function UseCaseCard({ id, title, description, index = 0 }: UseCaseCardProps) {
  return (
    <motion.article
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.1 }}
    >
      <div className="space-y-4">
        <p className="text-lg md:text-xl text-black leading-relaxed font-normal">
          {description}
        </p>
        <a
          href="#"
          className="text-black hover:opacity-70 underline transition-opacity inline-block text-base focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-1 -m-1"
          aria-label={`Explore ${title} use case`}
        >
          Explore use case
        </a>
      </div>
    </motion.article>
  );
}

