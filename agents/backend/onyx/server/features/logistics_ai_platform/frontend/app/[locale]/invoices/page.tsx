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
import { invoicesApi } from '@/lib/api/invoices';
import { formatCurrency, formatDate } from '@/lib/utils';
import { getStripe } from '@/lib/stripe';
import ErrorMessage from '@/components/ui/error-message';
import EmptyState from '@/components/ui/empty-state';
import { getErrorMessage } from '@/lib/error-handler';
import { CreditCard } from 'lucide-react';

const InvoicesPage = () => {
  const t = useTranslations();
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: invoices, isLoading, error } = useQuery({
    queryKey: ['invoices'],
    queryFn: () => invoicesApi.getAll(),
  });

  const handlePay = async (invoiceId: string, amount: number, currency: string) => {
    try {
      const stripe = await getStripe();
      if (!stripe) {
        alert('Stripe is not configured');
        return;
      }

      const response = await fetch('/api/stripe/create-checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          invoiceId,
          amount,
          currency: currency.toLowerCase(),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create checkout session');
      }

      const { sessionId } = await response.json();
      await stripe.redirectToCheckout({ sessionId });
    } catch (err) {
      console.error('Payment error:', err);
      alert('Failed to initiate payment. Please try again.');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">{t('invoices.title')}</h1>
              <p className="text-muted-foreground">View and pay your invoices</p>
            </div>
            <Link href={`/${locale}/invoices/create`}>
              <Button aria-label={t('invoices.createInvoice')}>{t('invoices.createInvoice')}</Button>
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
              <h1 className="text-3xl font-bold">{t('invoices.title')}</h1>
              <p className="text-muted-foreground">View and pay your invoices</p>
            </div>
            <Link href={`/${locale}/invoices/create`}>
              <Button aria-label={t('invoices.createInvoice')}>{t('invoices.createInvoice')}</Button>
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
            <h1 className="text-3xl font-bold">{t('invoices.title')}</h1>
            <p className="text-muted-foreground">View and pay your invoices</p>
          </div>
          <Link href={`/${locale}/invoices/create`}>
            <Button aria-label={t('invoices.createInvoice')}>{t('invoices.createInvoice')}</Button>
          </Link>
        </div>

        {!invoices || invoices.length === 0 ? (
          <EmptyState
            message="No invoices found."
            description="Invoices will appear here once created."
          />
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {invoices.map((invoice) => {
              const isPaid = invoice.status === 'paid';
              const isPending = invoice.status === 'pending';
              const isOverdue = invoice.due_date && new Date(invoice.due_date) < new Date() && !isPaid;

              return (
                <Card key={invoice.invoice_id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>{invoice.invoice_number}</CardTitle>
                      <Badge
                        variant={
                          isPaid ? 'success' : isOverdue ? 'destructive' : isPending ? 'warning' : 'default'
                        }
                      >
                        {invoice.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm text-muted-foreground">{t('invoices.total')}</p>
                        <p className="text-2xl font-bold">{formatCurrency(invoice.total, invoice.currency)}</p>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <p className="text-muted-foreground">Issue Date</p>
                          <p className="font-medium">{formatDate(invoice.issue_date)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">{t('invoices.dueDate')}</p>
                          <p className={`font-medium ${isOverdue ? 'text-destructive' : ''}`}>
                            {invoice.due_date ? formatDate(invoice.due_date) : 'N/A'}
                          </p>
                        </div>
                      </div>
                      {isPending && (
                        <Button
                          onClick={() => handlePay(invoice.invoice_id, invoice.total, invoice.currency)}
                          className="w-full"
                          aria-label={`Pay invoice ${invoice.invoice_number}`}
                        >
                          <CreditCard className="mr-2 h-4 w-4" />
                          {t('invoices.payNow')}
                        </Button>
                      )}
                      <Link href={`/${locale}/invoices/${invoice.invoice_id}`}>
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

export default InvoicesPage;
