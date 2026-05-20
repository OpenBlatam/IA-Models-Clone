'use client';

import { motion } from 'framer-motion';

interface DownloadButtonProps {
  label: string;
  ariaLabel: string;
  variant?: 'primary' | 'secondary';
  onClick?: () => void;
}

export function DownloadButton({ 
  label, 
  ariaLabel, 
  variant = 'primary',
  onClick 
}: DownloadButtonProps) {
  const className = variant === 'primary'
    ? 'bg-black text-white px-8 py-4 rounded-lg hover:bg-[#1a1a1a] transition-colors font-normal text-base focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2'
    : 'bg-white border border-[#000000] text-black px-8 py-4 rounded-lg hover:bg-[#f5f5f5] hover:border-[#000000] transition-colors font-normal text-base focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2';

  return (
    <motion.button
      className={className}
      aria-label={ariaLabel}
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      type="button"
    >
      {label}
    </motion.button>
  );
}

