'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';

interface BlogPostCardProps {
  id: string;
  date: string;
  category: string;
  title: string;
  ariaLabel: string;
  index?: number;
}

export function BlogPostCard({ id, date, category, title, ariaLabel, index = 0 }: BlogPostCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
    >
      <Link
        href={`/blog/${id}`}
        className="block space-y-3 hover:opacity-80 transition-opacity group focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-2 -m-2"
        aria-label={ariaLabel}
      >
        <div className="text-gray-400 text-sm font-normal">
          {date} • {category}
        </div>
        <h3 className="text-2xl md:text-3xl font-normal text-white group-hover:underline font-sans antialiased">
          {title}
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-white underline inline-block text-sm group-hover:opacity-70 transition-opacity">
            Read
          </span>
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </Link>
    </motion.article>
  );
}

