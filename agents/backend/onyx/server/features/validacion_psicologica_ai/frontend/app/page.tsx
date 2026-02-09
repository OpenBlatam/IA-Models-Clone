/**
 * Home page - Dashboard
 */

'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui';
import { ValidationList } from '@/components/features/ValidationList';
import { ValidationForm } from '@/components/features/ValidationForm';
import { DashboardStats } from '@/components/features/DashboardStats';
import { StatsChart } from '@/components/features/StatsChart';
import { QuickActions } from '@/components/features/QuickActions';
import { RecentValidations } from '@/components/features/RecentValidations';
import { PlatformStats } from '@/components/features/PlatformStats';
import { QuickStats } from '@/components/features/QuickStats';
import { ValidationMetrics } from '@/components/features/ValidationMetrics';
import { ActivityFeed } from '@/components/features/ActivityFeed';
import { PlatformInsights } from '@/components/features/PlatformInsights';
import { ValidationStats } from '@/components/features/ValidationStats';
import { ValidationProgress } from '@/components/features/ValidationProgress';
import { QuickStatsCards } from '@/components/features/QuickStatsCards';
import { RecentActivity } from '@/components/features/RecentActivity';
import { ValidationRating } from '@/components/features/ValidationRating';
import { ViewPreferences } from '@/components/features/ViewPreferences';
import { DashboardWidgets } from '@/components/features/DashboardWidgets';
import { CommandPalette } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { Brain, Settings, Compare, Plus, Search, Filter } from 'lucide-react';
import { NotificationCenter } from '@/components/features/NotificationCenter';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import type { CommandAction } from '@/components/ui/CommandPalette';

export default function HomePage() {
  const { data: validations } = useValidations();
  const router = useRouter();
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  const commandActions: CommandAction[] = [
    {
      id: 'new-validation',
      label: 'Nueva Validación',
      description: 'Crear una nueva validación psicológica',
      icon: <Plus className="h-4 w-4" />,
      action: () => {
        document.getElementById('validation-form')?.scrollIntoView({ behavior: 'smooth' });
        setIsCommandPaletteOpen(false);
      },
      category: 'Acciones',
    },
    {
      id: 'search',
      label: 'Buscar Validaciones',
      description: 'Buscar en todas las validaciones',
      icon: <Search className="h-4 w-4" />,
      action: () => {
        document.getElementById('validations-heading')?.scrollIntoView({ behavior: 'smooth' });
        setIsCommandPaletteOpen(false);
      },
      category: 'Navegación',
    },
    {
      id: 'connections',
      label: 'Gestionar Conexiones',
      description: 'Ver y gestionar conexiones de redes sociales',
      icon: <Settings className="h-4 w-4" />,
      action: () => {
        router.push('/connections');
        setIsCommandPaletteOpen(false);
      },
      category: 'Navegación',
    },
    {
      id: 'comparison',
      label: 'Comparar Validaciones',
      description: 'Comparar dos validaciones',
      icon: <Compare className="h-4 w-4" />,
      action: () => {
        router.push('/comparison');
        setIsCommandPaletteOpen(false);
      },
      category: 'Navegación',
    },
  ];

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-primary" aria-hidden="true" />
              <h1 className="text-2xl font-bold">Validación Psicológica AI</h1>
            </div>
            <nav className="flex items-center gap-4" role="navigation" aria-label="Navegación principal">
              <NotificationCenter />
              <Link href="/comparison" aria-label="Comparar validaciones">
                <Button variant="ghost" size="sm" tabIndex={0}>
                  <Compare className="h-4 w-4 mr-2" aria-hidden="true" />
                  Comparar
                </Button>
              </Link>
              <Link href="/connections" aria-label="Gestionar conexiones de redes sociales">
                <Button variant="ghost" size="sm" tabIndex={0}>
                  <Settings className="h-4 w-4 mr-2" aria-hidden="true" />
                  Conexiones
                </Button>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          <section aria-labelledby="quick-actions-heading">
            <h2 id="quick-actions-heading" className="sr-only">
              Acciones Rápidas
            </h2>
            <QuickActions />
          </section>

          <div className="grid lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-8">
          <section aria-labelledby="validation-stats-heading">
            <h2 id="validation-stats-heading" className="text-2xl font-bold mb-4">
              Estadísticas de Validaciones
            </h2>
            <ValidationStats />
          </section>

          <section aria-labelledby="validation-progress-heading">
            <h2 id="validation-progress-heading" className="text-2xl font-bold mb-4">
              Progreso de Validaciones
            </h2>
            <ValidationProgress />
          </section>

          <section aria-labelledby="dashboard-widgets-heading">
            <h2 id="dashboard-widgets-heading" className="text-2xl font-bold mb-4">
              Resumen del Dashboard
            </h2>
            <DashboardWidgets />
          </section>

          <section aria-labelledby="quick-stats-heading">
            <h2 id="quick-stats-heading" className="text-2xl font-bold mb-4">
              Estadísticas Rápidas
            </h2>
            <QuickStatsCards />
          </section>

          <section aria-labelledby="stats-heading">
            <h2 id="stats-heading" className="text-2xl font-bold mb-4">
              Estadísticas Detalladas
            </h2>
            <DashboardStats />
          </section>

          {validations && validations.length > 0 && (
            <>
              <section aria-labelledby="metrics-heading">
                <h2 id="metrics-heading" className="text-2xl font-bold mb-4">
                  Métricas Detalladas
                </h2>
                <ValidationMetrics />
              </section>

              <section aria-labelledby="chart-heading">
                <h2 id="chart-heading" className="text-2xl font-bold mb-4">
                  Tendencias
                </h2>
                <StatsChart validations={validations} type="line" />
              </section>

              <section aria-labelledby="insights-heading">
                <h2 id="insights-heading" className="text-2xl font-bold mb-4">
                  Insights por Plataforma
                </h2>
                <PlatformInsights />
              </section>
            </>
          )}

              <section aria-labelledby="validations-heading">
                <h2 id="validations-heading" className="text-2xl font-bold mb-4">
                  Validaciones
                </h2>
                <ValidationList />
              </section>
            </div>

            <aside className="space-y-6" aria-label="Panel lateral">
              <section aria-labelledby="new-validation-heading">
                <h2 id="new-validation-heading" className="sr-only">
                  Nueva Validación
                </h2>
                <ValidationForm />
              </section>

              <section aria-labelledby="recent-validations-heading">
                <h2 id="recent-validations-heading" className="sr-only">
                  Validaciones Recientes
                </h2>
                <RecentValidations />
              </section>

              <section aria-labelledby="platform-stats-heading">
                <h2 id="platform-stats-heading" className="sr-only">
                  Estadísticas de Plataformas
                </h2>
                <PlatformStats />
              </section>

              <section aria-labelledby="activity-heading">
                <h2 id="activity-heading" className="sr-only">
                  Actividad Reciente
                </h2>
                <RecentActivity />
              </section>

              <section aria-labelledby="rating-heading">
                <h2 id="rating-heading" className="sr-only">
                  Calificación General
                </h2>
                <ValidationRating />
              </section>

              <section aria-labelledby="preferences-heading">
                <h2 id="preferences-heading" className="sr-only">
                  Preferencias de Vista
                </h2>
                <ViewPreferences />
              </section>
            </aside>
          </div>
        </div>
      </main>

      <CommandPalette
        actions={commandActions}
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
      />
    </div>
  );
}

