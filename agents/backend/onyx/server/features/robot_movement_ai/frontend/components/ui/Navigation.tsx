'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface NavItem {
  id: string;
  label: string;
  href?: string;
  onClick?: () => void;
  children?: NavItem[];
  badge?: string;
}

interface NavigationProps {
  items: NavItem[];
  logo?: React.ReactNode;
  className?: string;
  sticky?: boolean;
  transparent?: boolean;
}

export default function Navigation({
  items,
  logo,
  className,
  sticky = true,
  transparent = false,
}: NavigationProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!sticky) return;
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [sticky]);

  const handleItemClick = (item: NavItem) => {
    if (item.onClick) {
      item.onClick();
    }
    if (!item.children) {
      setIsOpen(false);
    }
  };

  return (
    <nav
      ref={navRef}
      className={cn(
        'w-full z-50 transition-all duration-300',
        sticky && 'sticky top-0',
        scrolled || !transparent
          ? 'bg-white shadow-sm border-b border-gray-200'
          : 'bg-transparent',
        className
      )}
    >
      <div className="container-tesla mx-auto px-4 md:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-20">
          {/* Logo */}
          {logo && (
            <div className="flex-shrink-0">
              {logo}
            </div>
          )}

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-8">
            {items.map((item) => (
              <div key={item.id} className="relative group">
                <button
                  onClick={() => {
                    if (item.children) {
                      setActiveDropdown(activeDropdown === item.id ? null : item.id);
                    } else {
                      handleItemClick(item);
                    }
                  }}
                  className={cn(
                    'flex items-center gap-1 px-3 py-2 text-sm font-medium transition-colors min-h-[44px]',
                    scrolled || !transparent
                      ? 'text-tesla-black hover:text-tesla-blue'
                      : 'text-white hover:text-white/80'
                  )}
                >
                  {item.label}
                  {item.badge && (
                    <span className="ml-2 px-2 py-0.5 bg-tesla-blue text-white text-xs rounded-full">
                      {item.badge}
                    </span>
                  )}
                  {item.children && (
                    <ChevronDown className="w-4 h-4 transition-transform group-hover:rotate-180" />
                  )}
                </button>

                {/* Dropdown Menu */}
                {item.children && (
                  <AnimatePresence>
                    {activeDropdown === item.id && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full left-0 mt-2 w-64 bg-white rounded-lg shadow-tesla-lg border border-gray-200 p-2"
                      >
                        {item.children.map((child) => (
                          <button
                            key={child.id}
                            onClick={() => handleItemClick(child)}
                            className="w-full text-left px-4 py-3 rounded-md hover:bg-gray-50 transition-colors text-sm text-tesla-black min-h-[44px]"
                          >
                            <div className="flex items-center justify-between">
                              <span>{child.label}</span>
                              {child.badge && (
                                <span className="px-2 py-0.5 bg-tesla-blue text-white text-xs rounded-full">
                                  {child.badge}
                                </span>
                              )}
                            </div>
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                )}
              </div>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="lg:hidden p-2 text-tesla-black min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Toggle menu"
          >
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden bg-white border-t border-gray-200"
          >
            <div className="container-tesla mx-auto px-4 py-4 space-y-1">
              {items.map((item) => (
                <div key={item.id}>
                  <button
                    onClick={() => {
                      if (item.children) {
                        setActiveDropdown(activeDropdown === item.id ? null : item.id);
                      } else {
                        handleItemClick(item);
                      }
                    }}
                    className="w-full flex items-center justify-between px-4 py-3 text-left text-sm font-medium text-tesla-black hover:bg-gray-50 rounded-md transition-colors min-h-[44px]"
                  >
                    <span>{item.label}</span>
                    {item.children && (
                      <ChevronDown
                        className={cn(
                          'w-4 h-4 transition-transform',
                          activeDropdown === item.id && 'rotate-180'
                        )}
                      />
                    )}
                  </button>
                  {item.children && activeDropdown === item.id && (
                    <div className="pl-4 mt-1 space-y-1">
                      {item.children.map((child) => (
                        <button
                          key={child.id}
                          onClick={() => handleItemClick(child)}
                          className="w-full text-left px-4 py-2 text-sm text-tesla-gray-dark hover:bg-gray-50 rounded-md transition-colors min-h-[44px]"
                        >
                          {child.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}

