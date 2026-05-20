'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { SkeletonList } from '@/components/ui/skeleton';
import Navbar from '@/components/layout/navbar';
import { bookingsApi } from '@/lib/api/bookings';
import { formatDate } from '@/lib/utils';
import { getStatusBadgeVariant } from '@/lib/status-utils';
import ErrorMessage from '@/components/ui/error-message';
import EmptyState from '@/components/ui/empty-state';
import { getErrorMessage } from '@/lib/error-handler';

const BookingsPage = () => {
  const t = useTranslations('bookings');
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: bookings = [], isLoading, error } = useQuery({
    queryKey: ['bookings'],
    queryFn: () => bookingsApi.getAll(),
  });

  const renderHeader = () => (
    <div className="mb-8 flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">{t('title')}</h1>
        <p className="text-muted-foreground">Manage your bookings</p>
      </div>
      <Link href={`/${locale}/bookings/create`}>
        <Button aria-label={t('createBooking')}>{t('createBooking')}</Button>
      </Link>
    </div>
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          {renderHeader()}
          <SkeletonList count={5} />
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          {renderHeader()}
          <ErrorMessage message={getErrorMessage(error)} />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        {renderHeader()}

        {bookings.length === 0 ? (
          <EmptyState
            message="No bookings found."
            description="Create your first booking to get started."
            action={
              <Link href={`/${locale}/bookings/create`}>
                <Button>{t('createBooking')}</Button>
              </Link>
            }
          />
        ) : (
          <div className="space-y-4">
            {bookings.map((booking) => (
              <Card key={booking.booking_id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{booking.booking_reference}</CardTitle>
                      <CardDescription>
                        {t('bookingId')}: {booking.booking_id}
                      </CardDescription>
                    </div>
                    <Badge variant={getStatusBadgeVariant(booking.status)}>
                      {booking.status}
                    </Badge>
                  </div>
                </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                      <div>
                        <p className="text-sm text-muted-foreground">{t('estimatedDeparture')}</p>
                        <p className="font-medium">{formatDate(booking.estimated_departure)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">{t('estimatedArrival')}</p>
                        <p className="font-medium">{formatDate(booking.estimated_arrival)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Shipment ID</p>
                        <p className="font-medium">{booking.shipment_id}</p>
                      </div>
                      <div>
                        <Link
                          href={`/${locale}/bookings/${booking.booking_id}`}
                          aria-label={`View details for booking ${booking.booking_id}`}
                        >
                          <Button variant="outline" className="w-full">
                            View Details
                          </Button>
                        </Link>
                      </div>
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

export default BookingsPage;
