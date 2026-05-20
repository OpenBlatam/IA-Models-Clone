'use client';

import { useState } from 'react';
import { useTranslations } from 'next-intl';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import Navbar from '@/components/layout/navbar';
import { trackingApi } from '@/lib/api/tracking';
import { formatDateTime } from '@/lib/utils';
import { getStatusBadgeVariant } from '@/lib/status-utils';
import { Search, MapPin, Clock } from 'lucide-react';

const TrackingPage = () => {
  const t = useTranslations('tracking');
  const [identifier, setIdentifier] = useState('');
  const [trackingData, setTrackingData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleTrack = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!identifier.trim()) {
      setError('Please enter a tracking identifier');
      return;
    }

    setIsLoading(true);
    setError('');
    setTrackingData(null);

    try {
      const data = await trackingApi.track(identifier.trim());
      setTrackingData(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Tracking information not found');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isLoading) {
      handleTrack(e as any);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8" role="main">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">{t('title')}</h1>
          <p className="text-muted-foreground">{t('trackShipment')}</p>
        </div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>{t('enterTrackingNumber')}</CardTitle>
            <CardDescription>Enter tracking number, container number, or shipment ID</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleTrack} className="flex gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="TRK123456789, CONT1234567, or S12345678"
                  className="pl-10"
                  required
                  disabled={isLoading}
                  aria-label={t('enterTrackingNumber')}
                />
              </div>
              <Button type="submit" disabled={isLoading} aria-label="Track shipment">
                {isLoading ? 'Tracking...' : 'Track'}
              </Button>
            </form>
            {error && (
              <p className="mt-4 text-sm text-destructive" role="alert">
                {error}
              </p>
            )}
          </CardContent>
        </Card>

        {trackingData && trackingData.shipment && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Tracking Information</CardTitle>
                <Badge variant={getStatusBadgeVariant(trackingData.shipment.status)}>
                  {trackingData.shipment.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Tracking Number</p>
                    <p className="font-semibold">{trackingData.shipment.tracking_number || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Shipment Reference</p>
                    <p className="font-semibold">{trackingData.shipment.shipment_reference}</p>
                  </div>
                </div>

                {trackingData.shipment.tracking_events?.[0] && (
                  <div className="flex items-start gap-3 p-4 bg-muted rounded-lg">
                    <MapPin className="h-5 w-5 text-primary mt-0.5" />
                    <div>
                      <p className="font-medium">{t('currentLocation')}</p>
                      <p className="text-sm text-muted-foreground">
                        {trackingData.shipment.tracking_events[0].location?.city},{' '}
                        {trackingData.shipment.tracking_events[0].location?.country}
                      </p>
                    </div>
                  </div>
                )}

                {trackingData.shipment.estimated_arrival && (
                  <div className="flex items-start gap-3 p-4 bg-muted rounded-lg">
                    <Clock className="h-5 w-5 text-primary mt-0.5" />
                    <div>
                      <p className="font-medium">{t('estimatedArrival')}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatDateTime(trackingData.shipment.estimated_arrival)}
                      </p>
                    </div>
                  </div>
                )}

                {trackingData.shipment.tracking_events && trackingData.shipment.tracking_events.length > 0 && (
                  <div>
                    <p className="mb-4 text-sm font-medium">{t('trackingHistory')}</p>
                    <div className="space-y-4">
                      {trackingData.shipment.tracking_events.map((event: any, index: number) => (
                        <div key={index} className="border-l-2 border-primary pl-4 relative">
                          <div className="absolute -left-2 top-0 h-4 w-4 rounded-full bg-primary" />
                          <p className="font-medium">{event.description}</p>
                          <p className="text-sm text-muted-foreground">
                            {formatDateTime(event.timestamp)}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {event.location?.city}, {event.location?.country}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {!trackingData && !isLoading && !error && (
          <Card>
            <CardContent className="py-12 text-center">
              <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">{t('noTrackingInfo')}</p>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};

export default TrackingPage;
