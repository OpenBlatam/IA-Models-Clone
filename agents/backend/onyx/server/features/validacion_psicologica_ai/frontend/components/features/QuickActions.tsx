/**
 * Quick actions component for dashboard
 */

'use client';

import React from 'react';
import { Button, Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import Link from 'next/link';
import { Plus, Link2, Compare, FileText } from 'lucide-react';

export const QuickActions: React.FC = () => {
  const actions = [
    {
      title: 'Nueva Validación',
      description: 'Crear una nueva validación psicológica',
      icon: Plus,
      href: '/',
      variant: 'primary' as const,
    },
    {
      title: 'Conectar Red Social',
      description: 'Gestionar conexiones de redes sociales',
      icon: Link2,
      href: '/connections',
      variant: 'outline' as const,
    },
    {
      title: 'Comparar Validaciones',
      description: 'Comparar múltiples validaciones',
      icon: Compare,
      href: '/comparison',
      variant: 'outline' as const,
    },
  ];

  const handleKeyDown = (event: React.KeyboardEvent<HTMLAnchorElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      const href = (event.currentTarget as HTMLAnchorElement).getAttribute('href');
      if (href) {
        window.location.href = href;
      }
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Acciones Rápidas</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {actions.map((action) => {
            const Icon = action.icon;
            return (
              <Link
                key={action.title}
                href={action.href}
                onKeyDown={handleKeyDown}
                className="block"
                aria-label={action.description}
                tabIndex={0}
              >
                <Button
                  variant={action.variant}
                  className="w-full h-auto flex flex-col items-center justify-center gap-2 p-6"
                  tabIndex={-1}
                >
                  <Icon className="h-6 w-6" aria-hidden="true" />
                  <div className="text-center">
                    <div className="font-medium">{action.title}</div>
                    <div className="text-xs opacity-80 mt-1">{action.description}</div>
                  </div>
                </Button>
              </Link>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};




