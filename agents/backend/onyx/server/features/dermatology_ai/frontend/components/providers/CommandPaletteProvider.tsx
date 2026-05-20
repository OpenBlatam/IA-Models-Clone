'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { CommandPalette } from '../ui/CommandPalette';
import {
  Home,
  BarChart3,
  History,
  ShoppingBag,
  Settings,
  Compare,
  Bell,
} from 'lucide-react';

interface CommandPaletteProviderProps {
  children: React.ReactNode;
}

export const CommandPaletteProvider: React.FC<CommandPaletteProviderProps> = ({
  children,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  const commands = [
    {
      id: 'home',
      label: 'Ir al Inicio',
      description: 'Navegar a la página principal',
      icon: <Home className="h-4 w-4" />,
      action: () => router.push('/'),
      shortcut: 'Ctrl+H',
    },
    {
      id: 'dashboard',
      label: 'Ir al Dashboard',
      description: 'Ver estadísticas y métricas',
      icon: <BarChart3 className="h-4 w-4" />,
      action: () => router.push('/dashboard'),
      shortcut: 'Ctrl+D',
    },
    {
      id: 'history',
      label: 'Ver Historial',
      description: 'Revisar análisis anteriores',
      icon: <History className="h-4 w-4" />,
      action: () => router.push('/history'),
    },
    {
      id: 'compare',
      label: 'Comparar Análisis',
      description: 'Comparar dos análisis',
      icon: <Compare className="h-4 w-4" />,
      action: () => router.push('/compare'),
    },
    {
      id: 'products',
      label: 'Productos',
      description: 'Buscar productos de skincare',
      icon: <ShoppingBag className="h-4 w-4" />,
      action: () => router.push('/products'),
      shortcut: 'Ctrl+P',
    },
    {
      id: 'alerts',
      label: 'Alertas',
      description: 'Ver alertas y notificaciones',
      icon: <Bell className="h-4 w-4" />,
      action: () => router.push('/alerts'),
    },
    {
      id: 'settings',
      label: 'Configuración',
      description: 'Ajustes de la aplicación',
      icon: <Settings className="h-4 w-4" />,
      action: () => router.push('/settings'),
    },
  ];

  return (
    <>
      {children}
      <CommandPalette
        items={commands}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
      />
    </>
  );
};


