'use client';

import { motion } from 'framer-motion';

interface AnnouncementBannerProps {
  text: string;
  href?: string;
}

export function AnnouncementBanner({ text, href = '#' }: AnnouncementBannerProps) {
  return (
    <div className="mb-12 md:mb-16">
      <motion.a
        href={href}
        className="inline-flex items-center gap-2 text-black hover:opacity-70 transition-opacity group focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-2 -m-2"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <span className="text-base font-normal">{text}</span>
        <svg 
          className="w-5 h-5 group-hover:translate-x-1 transition-transform" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
      </motion.a>
    </div>
  );
}

