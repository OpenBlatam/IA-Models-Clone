'use client';

import { motion } from 'framer-motion';
import { Facebook, Twitter, Instagram, Linkedin, Mail, Phone, MapPin } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface FooterLink {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface FooterSection {
  title: string;
  links: FooterLink[];
}

interface FooterProps {
  sections?: FooterSection[];
  socialLinks?: {
    facebook?: string;
    twitter?: string;
    instagram?: string;
    linkedin?: string;
  };
  contact?: {
    email?: string;
    phone?: string;
    address?: string;
  };
  copyright?: string;
  className?: string;
}

export default function Footer({
  sections = [],
  socialLinks,
  contact,
  copyright = `© ${new Date().getFullYear()} Robot Movement AI. Todos los derechos reservados.`,
  className,
}: FooterProps) {
  const defaultSections: FooterSection[] = [
    {
      title: 'Producto',
      links: [
        { label: 'Características', href: '#features' },
        { label: 'Precios', href: '#pricing' },
        { label: 'Documentación', href: '#docs' },
      ],
    },
    {
      title: 'Soporte',
      links: [
        { label: 'Centro de Ayuda', href: '#help' },
        { label: 'Contacto', href: '#contact' },
        { label: 'Estado del Sistema', href: '#status' },
      ],
    },
    {
      title: 'Empresa',
      links: [
        { label: 'Acerca de', href: '#about' },
        { label: 'Blog', href: '#blog' },
        { label: 'Carreras', href: '#careers' },
      ],
    },
  ];

  const displaySections = sections.length > 0 ? sections : defaultSections;

  return (
    <footer className={cn('bg-tesla-black text-white', className)}>
      <div className="container-tesla mx-auto px-4 md:px-6 lg:px-8 py-12 md:py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 md:gap-12">
          {/* Brand Section */}
          <div className="lg:col-span-1">
            <motion.h3
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-2xl font-bold mb-4"
            >
              Robot Movement AI
            </motion.h3>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="text-gray-400 text-sm mb-6"
            >
              Plataforma IA de Movimiento Robótico de última generación.
            </motion.p>

            {/* Social Links */}
            {socialLinks && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
                className="flex items-center gap-4"
              >
                {socialLinks.facebook && (
                  <a
                    href={socialLinks.facebook}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                    aria-label="Facebook"
                  >
                    <Facebook className="w-5 h-5" />
                  </a>
                )}
                {socialLinks.twitter && (
                  <a
                    href={socialLinks.twitter}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                    aria-label="Twitter"
                  >
                    <Twitter className="w-5 h-5" />
                  </a>
                )}
                {socialLinks.instagram && (
                  <a
                    href={socialLinks.instagram}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                    aria-label="Instagram"
                  >
                    <Instagram className="w-5 h-5" />
                  </a>
                )}
                {socialLinks.linkedin && (
                  <a
                    href={socialLinks.linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                    aria-label="LinkedIn"
                  >
                    <Linkedin className="w-5 h-5" />
                  </a>
                )}
              </motion.div>
            )}
          </div>

          {/* Links Sections */}
          {displaySections.map((section, index) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 * (index + 1) }}
            >
              <h4 className="text-sm font-semibold mb-4 uppercase tracking-wide">
                {section.title}
              </h4>
              <ul className="space-y-3">
                {section.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      onClick={link.onClick}
                      className="text-gray-400 hover:text-white text-sm transition-colors"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}

          {/* Contact Section */}
          {contact && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
            >
              <h4 className="text-sm font-semibold mb-4 uppercase tracking-wide">
                Contacto
              </h4>
              <ul className="space-y-3">
                {contact.email && (
                  <li className="flex items-center gap-3 text-gray-400 text-sm">
                    <Mail className="w-4 h-4" />
                    <a href={`mailto:${contact.email}`} className="hover:text-white transition-colors">
                      {contact.email}
                    </a>
                  </li>
                )}
                {contact.phone && (
                  <li className="flex items-center gap-3 text-gray-400 text-sm">
                    <Phone className="w-4 h-4" />
                    <a href={`tel:${contact.phone}`} className="hover:text-white transition-colors">
                      {contact.phone}
                    </a>
                  </li>
                )}
                {contact.address && (
                  <li className="flex items-start gap-3 text-gray-400 text-sm">
                    <MapPin className="w-4 h-4 mt-0.5" />
                    <span>{contact.address}</span>
                  </li>
                )}
              </ul>
            </motion.div>
          )}
        </div>

        {/* Copyright */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mt-12 pt-8 border-t border-white/10 text-center text-gray-400 text-sm"
        >
          {copyright}
        </motion.div>
      </div>
    </footer>
  );
}



