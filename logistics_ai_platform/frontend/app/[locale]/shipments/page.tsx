'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { SkeletonList } from '@/components/ui/skeleton';
import Navbar from '@/components/layout/navbar';
import { shipmentsApi } from '@/lib/api/shipments';
import { formatDate } from '@/lib/utils';
import { getStatusBadgeVariant } from '@/lib/status-utils';
import ErrorMessage from '@/components/ui/error-message';
import EmptyState from '@/components/ui/empty-state';
import { getErrorMessage } from '@/lib/error-handler';

const ShipmentsPage = () => {
  const t = useTranslations();
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: shipments, isLoading, error } = useQuery({
    queryKey: ['shipments'],
    queryFn: () => shipmentsApi.getAll(),
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">{t('shipments.title')}</h1>
              <p className="text-muted-foreground">Track and manage your shipments</p>
            </div>
            <Link href={`/${locale}/shipments/create`}>
              <Button aria-label={t('shipments.createShipment')}>{t('shipments.createShipment')}</Button>
            </Link>
          </div>
          <SkeletonList count={6} />
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">{t('shipments.title')}</h1>
              <p className="text-muted-foreground">Track and manage your shipments</p>
            </div>
            <Link href={`/${locale}/shipments/create`}>
              <Button aria-label={t('shipments.createShipment')}>{t('shipments.createShipment')}</Button>
            </Link>
          </div>
          <ErrorMessage message={getErrorMessage(error)} />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">{t('shipments.title')}</h1>
            <p className="text-muted-foreground">Track and manage your shipments</p>
          </div>
          <Link href={`/${locale}/shipments/create`}>
            <Button aria-label={t('shipments.createShipment')}>{t('shipments.createShipment')}</Button>
          </Link>
        </div>

        {!shipments || shipments.length === 0 ? (
          <EmptyState
            message="No shipments found."
            description="Create your first shipment to get started."
            action={
              <Link href={`/${locale}/shipments/create`}>
                <Button>{t('shipments.createShipment')}</Button>
              </Link>
            }
          />
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {shipments.map((shipment) => (
              <Card key={shipment.shipment_id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>{shipment.shipment_reference}</CardTitle>
                    <Badge variant={getStatusBadgeVariant(shipment.status)}>
                      {shipment.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      <span className="font-medium">{t('shipments.origin')}:</span> {shipment.origin.city},{' '}
                      {shipment.origin.country}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      <span className="font-medium">{t('shipments.destination')}:</span> {shipment.destination.city},{' '}
                      {shipment.destination.country}
                    </p>
                    {shipment.tracking_number && (
                      <p className="text-sm text-muted-foreground">
                        <span className="font-medium">{t('shipments.trackingNumber')}:</span> {shipment.tracking_number}
                      </p>
                    )}
                    <Link
                      href={`/${locale}/shipments/${shipment.shipment_id}`}
                      className="mt-4 block"
                      aria-label={`View details for shipment ${shipment.shipment_id}`}
                    >
                      <Button variant="outline" className="w-full">
                        {t('common.view')}
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default ShipmentsPage;
