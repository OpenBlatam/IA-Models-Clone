'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Navbar from '@/components/layout/navbar';
import { shipmentsApi } from '@/lib/api/shipments';
import { alertsApi } from '@/lib/api/alerts';
import { SkeletonCard } from '@/components/ui/skeleton';
import Loading from '@/components/ui/loading';
import ErrorMessage from '@/components/ui/error-message';
import { getErrorMessage } from '@/lib/error-handler';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';

const DashboardPage = () => {
  const t = useTranslations();
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: shipments, isLoading: shipmentsLoading, error: shipmentsError } = useQuery({
    queryKey: ['shipments'],
    queryFn: () => shipmentsApi.getAll({ limit: 10 }),
  });

  const { data: alerts, isLoading: alertsLoading, error: alertsError } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => alertsApi.getAll(),
  });

  const isLoading = shipmentsLoading || alertsLoading;
  const hasError = shipmentsError || alertsError;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('dashboard.title')}</h1>
            <p className="text-muted-foreground">{t('dashboard.overview')}</p>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
        </main>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('dashboard.title')}</h1>
            <p className="text-muted-foreground">{t('dashboard.overview')}</p>
          </div>
          <ErrorMessage message={getErrorMessage(shipmentsError || alertsError)} />
        </main>
      </div>
    );
  }

  const activeShipments = shipments?.filter((s) => s.status === 'in_transit').length || 0;
  const unreadAlerts = alerts?.filter((a) => !a.is_read).length || 0;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">{t('dashboard.title')}</h1>
          <p className="text-muted-foreground">{t('dashboard.overview')}</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{t('dashboard.activeShipments')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{activeShipments}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {shipments?.length || 0} total shipments
              </p>
              <Link href={`/${locale}/shipments`} className="mt-4 inline-block">
                <Button variant="ghost" size="sm" className="h-8 px-2">
                  View all <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{t('dashboard.pendingQuotes')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">0</div>
              <p className="text-xs text-muted-foreground mt-1">Pending review</p>
              <Link href={`/${locale}/quotes`} className="mt-4 inline-block">
                <Button variant="ghost" size="sm" className="h-8 px-2">
                  View all <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{t('dashboard.recentBookings')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">0</div>
              <p className="text-xs text-muted-foreground mt-1">Last 30 days</p>
              <Link href={`/${locale}/bookings`} className="mt-4 inline-block">
                <Button variant="ghost" size="sm" className="h-8 px-2">
                  View all <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{t('dashboard.alerts')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{unreadAlerts}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {alerts?.length || 0} total alerts
              </p>
              <Link href={`/${locale}/alerts`} className="mt-4 inline-block">
                <Button variant="ghost" size="sm" className="h-8 px-2">
                  View all <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {unreadAlerts > 0 && (
          <Card className="mt-8">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{t('alerts.title')}</CardTitle>
                <Link href={`/${locale}/alerts`}>
                  <Button variant="ghost" size="sm">
                    View all <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {alerts
                  ?.filter((alert) => !alert.is_read)
                  .slice(0, 5)
                  .map((alert) => (
                    <div key={alert.alert_id} className="border-b pb-2 last:border-b-0">
                      <p className="font-medium">{alert.message}</p>
                      <p className="text-sm text-muted-foreground">{alert.alert_type}</p>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};

export default DashboardPage;
