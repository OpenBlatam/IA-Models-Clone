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
import { quotesApi } from '@/lib/api/quotes';
import type { QuoteResponse } from '@/types/api';
import { formatCurrency, formatDate } from '@/lib/utils';
import ErrorMessage from '@/components/ui/error-message';
import EmptyState from '@/components/ui/empty-state';
import { getErrorMessage } from '@/lib/error-handler';

const QuotesPage = () => {
  const t = useTranslations('quotes');
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: quotes = [], isLoading, error } = useQuery({
    queryKey: ['quotes'],
    queryFn: () => quotesApi.getAll(),
  });

  const renderHeader = () => (
    <div className="mb-8 flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">{t('title')}</h1>
        <p className="text-muted-foreground">Manage your freight quotes</p>
      </div>
      <Link href={`/${locale}/quotes/create`}>
        <Button aria-label={t('createQuote')}>{t('createQuote')}</Button>
      </Link>
    </div>
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          {renderHeader()}
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

        {quotes.length === 0 ? (
          <EmptyState
            message="No quotes found. Create your first quote to get started."
            action={
              <Link href={`/${locale}/quotes/create`}>
                <Button>{t('createQuote')}</Button>
              </Link>
            }
          />
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {quotes.map((quote) => {
              const firstOption = quote.options?.[0];
              const originText = `${quote.origin.city}, ${quote.origin.country}`;
              const destinationText = `${quote.destination.city}, ${quote.destination.country}`;
              const isValid = new Date(quote.valid_until) > new Date();

              return (
                <Card key={quote.quote_id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{quote.quote_id}</CardTitle>
                      <Badge variant={isValid ? 'success' : 'destructive'}>
                        {isValid ? 'Valid' : 'Expired'}
                      </Badge>
                    </div>
                    <CardDescription>
                      {originText} → {destinationText}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {firstOption && (
                        <>
                          <p className="text-sm">
                            <span className="font-medium">{t('transportationMode')}:</span>{' '}
                            <Badge variant="outline" className="ml-1">
                              {firstOption.transportation_mode}
                            </Badge>
                          </p>
                          <p className="text-sm">
                            <span className="font-medium">{t('price')}:</span>{' '}
                            <span className="font-semibold text-lg">
                              {formatCurrency(firstOption.price_usd || 0, firstOption.currency)}
                            </span>
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {firstOption.transit_days} transit days
                          </p>
                        </>
                      )}
                      <p className="text-sm">
                        <span className="font-medium">{t('validUntil')}:</span> {formatDate(quote.valid_until)}
                      </p>
                      <Link
                        href={`/${locale}/quotes/${quote.quote_id}`}
                        className="mt-4 block"
                        aria-label={`View details for quote ${quote.quote_id}`}
                      >
                        <Button variant="outline" className="w-full">
                          View Details
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
};

export default QuotesPage;
