'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { Logo } from './Logo';
import { NAV_ITEMS, MOBILE_MENU_ITEMS } from './constants';

export function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  const mobileMenuRef = useRef<HTMLElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  const toggleMenu = useCallback(() => {
    setMenuOpen((prev) => !prev);
  }, []);

  const closeMenu = useCallback(() => {
    setMenuOpen(false);
  }, []);

  // Close menu on escape key and handle focus trap
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && menuOpen) {
        closeMenu();
        menuButtonRef.current?.focus();
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (
        menuOpen &&
        mobileMenuRef.current &&
        !mobileMenuRef.current.contains(e.target as Node) &&
        !menuButtonRef.current?.contains(e.target as Node)
      ) {
        closeMenu();
      }
    };

    if (menuOpen) {
      document.addEventListener('keydown', handleEscape);
      document.addEventListener('mousedown', handleClickOutside);
      document.body.style.overflow = 'hidden';
      const firstFocusable = mobileMenuRef.current?.querySelector<HTMLElement>(
        'button, a, [tabindex]:not([tabindex="-1"])'
      );
      firstFocusable?.focus();
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('mousedown', handleClickOutside);
      document.body.style.overflow = '';
    };
  }, [menuOpen, closeMenu]);

  return (
    <motion.header 
      className="relative z-10 bg-white"
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8 py-3.5 md:py-4">
        <div className="flex items-center justify-between">
          <motion.a 
            href="/" 
            className="flex items-center gap-2.5 text-base text-black hover:opacity-80 transition-opacity duration-200 ease-in-out no-underline font-normal leading-normal"
            aria-label="Home - bulk"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
          >
            <Logo size="sm" showText={true} gradientId="gradient-header" />
          </motion.a>
          
          <nav className="hidden md:flex items-center gap-7 lg:gap-8" aria-label="Main navigation">
            {NAV_ITEMS.map((item, index) => (
              <motion.button
                key={item.label}
                className="text-black hover:opacity-70 transition-opacity duration-200 ease-in-out font-normal text-sm leading-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
                aria-label={item.ariaLabel}
                onClick={item.onClick}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  delay: index * 0.1, 
                  duration: 0.3,
                  type: "spring",
                  stiffness: 300,
                  damping: 20
                }}
                whileHover={{ opacity: 0.7 }}
                whileTap={{ scale: 0.98 }}
              >
                {item.label}
              </motion.button>
            ))}
            <motion.a 
              href="#overview" 
              className="text-black hover:opacity-70 transition-opacity duration-200 ease-in-out font-normal text-sm leading-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded no-underline"
              aria-label="See overview"
              whileHover={{ opacity: 0.7 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: "spring", stiffness: 400, damping: 17 }}
            >
              See overview
            </motion.a>
            <motion.button 
              className="bg-black text-white px-4 py-2 rounded-md hover:bg-[#1a1a1a] focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 transition-colors duration-200 ease-in-out font-normal text-sm leading-normal flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
              aria-label="Download bulk"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: "spring", stiffness: 400, damping: 17 }}
              type="button"
            >
              Download
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </motion.button>
          </nav>

          <motion.button 
            ref={menuButtonRef}
            className={clsx(
              "md:hidden text-black hover:opacity-70 transition-opacity duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded",
              menuOpen && "opacity-70"
            )}
            onClick={toggleMenu}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
            aria-controls="mobile-menu"
            whileTap={{ scale: 0.95 }}
            animate={{ rotate: menuOpen ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              {menuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </motion.button>
        </div>

        <AnimatePresence>
          {menuOpen && (
            <motion.nav 
              ref={mobileMenuRef}
              id="mobile-menu"
              className="md:hidden mt-4 space-y-2 pb-4"
              aria-label="Mobile navigation"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2, ease: "easeInOut" }}
            >
              {MOBILE_MENU_ITEMS.map((item, index) => (
                <motion.button
                  key={item.label}
                  className="block w-full text-left text-black hover:opacity-70 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded py-2 text-sm transition-opacity duration-200 ease-in-out"
                  aria-label={item.ariaLabel}
                  onClick={() => {
                    item.onClick?.();
                    closeMenu();
                  }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05, duration: 0.2 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {item.label}
                </motion.button>
              ))}
              <div className="pt-2 pb-2 space-y-3 border-t border-gray-200 mt-2">
                <div className="text-black text-sm font-normal">una ia que no para agentes que no paran</div>
                <div className="text-black text-sm font-normal">Explora cómo bulk funciona para ti</div>
                <a 
                  href="#overview" 
                  className="block text-black hover:opacity-70 underline transition-opacity duration-200 ease-in-out text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
                  onClick={closeMenu}
                  aria-label="See overview"
                >
                  See overview
                </a>
              </div>
              <motion.button 
                className="block w-full text-left bg-black text-white px-4 py-2 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 transition-colors duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={closeMenu}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="button"
                aria-label="Download bulk"
              >
                Download
              </motion.button>
            </motion.nav>
          )}
        </AnimatePresence>
      </div>
    </motion.header>
  );
}

