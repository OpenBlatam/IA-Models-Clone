'use client';

import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { Logo } from './Logo';

export function HeroSection() {
  return (
    <motion.section
      className="pt-20 md:pt-28 lg:pt-36 pb-28 md:pb-36 lg:pb-44 relative z-10"
      aria-label="Hero section"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.2 }}
    >
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-center text-center">
          {/* Large Logo - centered - exact antigravity spacing */}
          <motion.div
            className="mb-20 md:mb-24 lg:mb-28"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{
              duration: 0.5,
              delay: 0.3,
              type: "spring",
              stiffness: 100,
              damping: 20
            }}
          >
            <Logo size="lg" showText={true} gradientId="gradient-hero" />
          </motion.div>

          <motion.h1
            className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-normal mb-8 md:mb-10 text-black leading-[1.08] tracking-[-0.02em] md:tracking-[-0.03em] font-sans antialiased max-w-5xl mx-auto"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{
              duration: 0.6,
              delay: 0.4,
              type: "spring",
              stiffness: 100,
              damping: 20
            }}
          >
            bulk
          </motion.h1>
          <motion.p
            className="text-lg md:text-xl lg:text-2xl text-[#2a2a2a] mb-14 md:mb-20 font-normal font-sans antialiased leading-[1.4] tracking-[-0.008em] md:tracking-[-0.01em] max-w-3xl mx-auto"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{
              duration: 0.6,
              delay: 0.5,
              type: "spring",
              stiffness: 100,
              damping: 20
            }}
          >
            una ia que no para agentes que no paran
          </motion.p>
        </div>

        <motion.div
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-10"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{
            duration: 0.6,
            delay: 0.6,
            type: "spring",
            stiffness: 100,
            damping: 20
          }}
        >
          <motion.a
            href="/agent-control"
            className={clsx(
              "bg-[#000000] text-[#ffffff] px-8 py-3.5 rounded-lg",
              "hover:bg-[#1a1a1a] active:bg-[#0a0a0a] focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2",
              "transition-all duration-200 ease-in-out font-normal flex items-center justify-center gap-2",
              "text-base leading-normal disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap no-underline"
            )}
            aria-label="Open Agent Control"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
          >
            <svg
              className="w-5 h-5 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 24 24"
              preserveAspectRatio="xMidYMid meet"
              aria-hidden="true"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z" />
            </svg>
            Open Agent Control
          </motion.a>
          <motion.button
            className={clsx(
              "bg-white border border-[#000000] text-[#000000] px-8 py-3.5 rounded-lg",
              "hover:bg-[#f5f5f5] hover:border-[#000000] active:bg-[#eeeeee]",
              "focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2",
              "transition-all duration-200 ease-in-out font-normal text-base leading-normal whitespace-nowrap",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            aria-label="Explore use cases"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
            type="button"
          >
            Explore use cases
          </motion.button>
        </motion.div>
      </div>
    </motion.section>
  );
}

