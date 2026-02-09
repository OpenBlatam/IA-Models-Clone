'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';

interface FooterLink {
  id: string;
  label: string;
  href: string;
}

const FOOTER_LINKS_LEFT: FooterLink[] = [
  { id: 'download', label: 'Download', href: '/download' },
  { id: 'product', label: 'Product', href: '/product' },
  { id: 'doc', label: 'Doc', href: '#' },
  { id: 'changelog', label: 'Changelog', href: '#' },
];

const FOOTER_LINKS_RIGHT: FooterLink[] = [
  { id: 'blog', label: 'Blog', href: '/blog' },
  { id: 'pricing', label: 'Pricing', href: '/pricing' },
  { id: 'use-cases', label: 'Use Cases', href: '/use-cases' },
];

const FOOTER_NAV_LINKS: FooterLink[] = [
  { id: 'about', label: 'About bulk', href: '#' },
  { id: 'products', label: 'bulk Product', href: '#' },
  { id: 'privacy', label: 'Privacy', href: '#' },
  { id: 'terms', label: 'Term', href: '#' },
];

export function Footer() {
  const leftLinks = useMemo(() => FOOTER_LINKS_LEFT, []);
  const rightLinks = useMemo(() => FOOTER_LINKS_RIGHT, []);
  const navLinks = useMemo(() => FOOTER_NAV_LINKS, []);

  return (
    <footer className="border-t border-gray-200 py-12 md:py-16 relative z-10" role="contentinfo">
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-12 md:gap-16 mb-12 md:mb-16">
          <nav className="space-y-4" aria-label="Footer navigation - Product">
            <div className="space-y-3">
              {leftLinks.map((link) => (
                <a 
                  key={link.id}
                  href={link.href} 
                  className="block text-black hover:opacity-70 transition-opacity duration-200 ease-in-out text-sm leading-normal font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </nav>
          <nav className="space-y-4" aria-label="Footer navigation - Resources">
            <div className="space-y-3">
              {rightLinks.map((link) => (
                <a 
                  key={link.id}
                  href={link.href} 
                  className="block text-black hover:opacity-70 transition-opacity duration-200 ease-in-out text-sm leading-normal font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </nav>
        </div>
        
        <div className="border-t border-gray-200 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
            <a 
              href="/" 
              className="text-black hover:opacity-70 transition-opacity duration-200 ease-in-out text-lg font-normal leading-normal tracking-[-0.01em] focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
              aria-label="Home - bulk"
            >
              bulk
            </a>
            <nav className="flex flex-wrap gap-6 text-sm" aria-label="Footer legal navigation">
              {navLinks.map((link) => (
                <a 
                  key={link.id}
                  href={link.href} 
                  className="text-black hover:opacity-70 transition-opacity duration-200 ease-in-out text-sm leading-normal font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
                >
                  {link.label}
                </a>
              ))}
            </nav>
          </div>
        </div>

      </div>
    </footer>
  );
}

