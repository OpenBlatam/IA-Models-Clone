'use client';

import Link from 'next/link';
import { Home, History, Search, Heart, BarChart3 } from 'lucide-react';
import { ActiveLink } from './navigation/active-link';

const navItems = [
  { href: '/', label: 'Inicio', icon: Home, exact: true },
  { href: '/history', label: 'Historial', icon: History },
  { href: '/search', label: 'Búsqueda', icon: Search },
  { href: '/favorites', label: 'Favoritos', icon: Heart },
  { href: '/analytics', label: 'Analytics', icon: BarChart3 },
];

export const Navigation = (): JSX.Element => {
  return (
    <nav className="bg-white shadow-sm border-b" role="navigation" aria-label="Navegación principal">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link 
            href="/" 
            className="flex items-center space-x-2 focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-offset-2 rounded"
            aria-label="Ir al inicio"
          >
            <span className="text-2xl font-bold text-indigo-600">
              Manuales Hogar AI
            </span>
          </Link>
          <div className="flex items-center space-x-4" role="menubar">
            {navItems.map((item) => (
              <ActiveLink
                key={item.href}
                href={item.href}
                label={item.label}
                icon={item.icon}
                exact={item.exact}
              />
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

