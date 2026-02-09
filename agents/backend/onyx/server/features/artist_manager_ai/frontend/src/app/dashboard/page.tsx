'use client';

import { useDashboard, useDailySummary } from '@/hooks/use-dashboard';
import { useArtist } from '@/hooks/use-artist';
import { PageLayout, PageHeader } from '@/components/layout';
import { QueryWrapper, StatsGrid, Card, CardHeader, CardTitle, CardContent, Alert, EventList, StatsList } from '@/components/ui';
import { Calendar, Clock, AlertCircle, Shirt, TrendingUp } from 'lucide-react';
import Link from 'next/link';

const DashboardPage = () => {
  const { artistId } = useArtist();
  const dashboardQuery = useDashboard(artistId);
  const summaryQuery = useDailySummary(artistId);

  return (
    <PageLayout>
      <PageHeader title="Dashboard" />

      <QueryWrapper query={summaryQuery}>
        {(summary) => (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Resumen del Día</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 mb-4">{summary.summary}</p>
              {summary.motivation && (
                <Alert variant="info" message={summary.motivation} />
              )}
            </CardContent>
          </Card>
        )}
      </QueryWrapper>

      <QueryWrapper query={dashboardQuery}>
        {(dashboard) => (
          <>
            <StatsGrid
              stats={[
                {
                  label: 'Eventos Próximos',
                  value: dashboard.upcoming_events.count,
                  icon: <Calendar className="w-5 h-5" />,
                  variant: 'primary',
                },
                {
                  label: 'Rutinas Pendientes',
                  value: dashboard.routines.pending_count,
                  icon: <Clock className="w-5 h-5" />,
                  variant: 'warning',
                },
                {
                  label: 'Protocolos Críticos',
                  value: dashboard.protocols.critical_count,
                  icon: <AlertCircle className="w-5 h-5" />,
                  variant: 'danger',
                },
                {
                  label: 'Items en Guardarropa',
                  value: dashboard.wardrobe.total_items,
                  icon: <Shirt className="w-5 h-5" />,
                  variant: 'info',
                },
              ]}
              columns={4}
              className="mb-6"
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Próximos Eventos</CardTitle>
                    <Link href="/calendar" className="text-sm text-blue-600 hover:text-blue-700">
                      Ver todos
                    </Link>
                  </div>
                </CardHeader>
                <CardContent>
                  <EventList
                    events={dashboard.upcoming_events.events}
                    maxItems={5}
                    emptyMessage="No hay eventos próximos"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Estadísticas
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <StatsList
                    items={[
                      {
                        label: 'Rutinas Completadas Hoy',
                        value: dashboard.routines.completed_today,
                        variant: 'success',
                      },
                      {
                        label: 'Total de Rutinas',
                        value: dashboard.routines.total,
                        variant: 'primary',
                      },
                      {
                        label: 'Total de Protocolos',
                        value: dashboard.protocols.total,
                        variant: 'info',
                      },
                      {
                        label: 'Total de Outfits',
                        value: dashboard.wardrobe.total_outfits,
                        variant: 'default',
                      },
                    ]}
                  />
                </CardContent>
              </Card>
            </div>
          </>
        )}
      </QueryWrapper>
    </PageLayout>
  );
};

export default DashboardPage;

