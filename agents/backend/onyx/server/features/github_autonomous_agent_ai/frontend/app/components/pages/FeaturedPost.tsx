'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';

interface FeaturedPostProps {
  id: string;
  title: string;
  ariaLabel: string;
}

export function FeaturedPost({ id, title, ariaLabel }: FeaturedPostProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <Link
        href={`/blog/${id}`}
        className="block group focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-2 -m-2"
        aria-label={ariaLabel}
      >
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-normal text-white mb-6 group-hover:opacity-80 transition-opacity font-sans antialiased">
          {title}
        </h1>
        <div className="flex items-center gap-2">
          <span className="text-white underline inline-block text-base group-hover:opacity-70 transition-opacity">
            Read
          </span>
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </Link>
    </motion.article>
  );
}

