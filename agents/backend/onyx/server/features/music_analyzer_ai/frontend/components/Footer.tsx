/**
 * Footer component for the application.
 * Provides footer with links, copyright, and social media.
 * Enhanced with accessibility and responsive design.
 */

'use client';

import Link from 'next/link';
import { Music, Bot, Github, Twitter, Mail } from 'lucide-react';
import { ROUTES } from '@/lib/constants';
import { appConfig } from '@/lib/config/app';

/**
 * Footer link interface.
 */
interface FooterLink {
  href: string;
  label: string;
  external?: boolean;
}

/**
 * Footer section interface.
 */
interface FooterSection {
  title: string;
  links: FooterLink[];
}

/**
 * Footer sections configuration.
 */
const FOOTER_SECTIONS: FooterSection[] = [
  {
    title: 'Plataformas',
    links: [
      { href: ROUTES.MUSIC, label: 'Music Analyzer AI' },
      { href: ROUTES.ROBOT, label: 'Robot Movement AI' },
    ],
  },
  {
    title: 'Recursos',
    links: [
      { href: '/docs', label: 'Documentación' },
      { href: '/api', label: 'API' },
      { href: '/support', label: 'Soporte' },
    ],
  },
  {
    title: 'Legal',
    links: [
      { href: '/privacy', label: 'Privacidad' },
      { href: '/terms', label: 'Términos' },
      { href: '/cookies', label: 'Cookies' },
    ],
  },
] as const;

/**
 * Social media links.
 */
const SOCIAL_LINKS = [
  {
    href: 'https://github.com/blatam-academy',
    label: 'GitHub',
    icon: Github,
  },
  {
    href: 'https://twitter.com/blatamacademy',
    label: 'Twitter',
    icon: Twitter,
  },
  {
    href: 'mailto:contact@blatam-academy.com',
    label: 'Email',
    icon: Mail,
  },
] as const;

/**
 * Footer component.
 * Displays footer with navigation links and social media.
 *
 * @returns Footer component
 */
export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer
      className="bg-slate-900/50 backdrop-blur-lg border-t border-white/10 mt-auto"
      role="contentinfo"
    >
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          {/* Brand Section */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Music className="w-6 h-6 text-purple-400" aria-hidden="true" />
              <h3 className="text-xl font-bold text-white">
                {appConfig.name}
              </h3>
            </div>
            <p className="text-gray-400 text-sm mb-4">
              {appConfig.description}
            </p>
            <div className="flex gap-4">
              {SOCIAL_LINKS.map((social) => {
                const Icon = social.icon;
                return (
                  <a
                    key={social.href}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-purple-400 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-slate-900 rounded"
                    aria-label={social.label}
                  >
                    <Icon className="w-5 h-5" aria-hidden="true" />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Footer Sections */}
          {FOOTER_SECTIONS.map((section) => (
            <div key={section.title}>
              <h4 className="text-white font-semibold mb-4">{section.title}</h4>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link.href}>
                    {link.external ? (
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-400 hover:text-purple-400 transition-colors text-sm focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-slate-900 rounded"
                      >
                        {link.label}
                      </a>
                    ) : (
                      <Link
                        href={link.href}
                        className="text-gray-400 hover:text-purple-400 transition-colors text-sm focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-slate-900 rounded"
                      >
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Copyright */}
        <div className="border-t border-white/10 pt-8">
          <p className="text-center text-gray-400 text-sm">
            © {currentYear} {appConfig.author}. Todos los derechos reservados.
          </p>
        </div>
      </div>
    </footer>
  );
}

