'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Navbar from '@/components/layout/navbar';
import { insuranceApi } from '@/lib/api/insurance';
import { formatCurrency, formatDate } from '@/lib/utils';

const InsurancePage = () => {
  const t = useTranslations('insurance');
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: policies = [], isLoading, error } = useQuery({
    queryKey: ['insurance'],
    queryFn: async () => {
      try {
        return [];
      } catch (err) {
        console.error('Error loading insurance policies:', err);
        return [];
      }
    },
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">{t('title')}</h1>
              <p className="text-muted-foreground">Manage insurance policies</p>
            </div>
            <Link href={`/${locale}/insurance/create`}>
              <Button aria-label={t('createInsurance')}>{t('createInsurance')}</Button>
            </Link>
          </div>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">Loading...</p>
            </CardContent>
          </Card>
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
              <h1 className="text-3xl font-bold">{t('title')}</h1>
              <p className="text-muted-foreground">Manage insurance policies</p>
            </div>
            <Link href={`/${locale}/insurance/create`}>
              <Button aria-label={t('createInsurance')}>{t('createInsurance')}</Button>
            </Link>
          </div>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-destructive">Error loading insurance policies. Please try again later.</p>
            </CardContent>
          </Card>
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
            <h1 className="text-3xl font-bold">{t('title')}</h1>
            <p className="text-muted-foreground">Manage insurance policies</p>
          </div>
          <Link href={`/${locale}/insurance/create`}>
            <Button aria-label={t('createInsurance')}>{t('createInsurance')}</Button>
          </Link>
        </div>

        {policies.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">No insurance policies found.</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {policies.map((policy) => (
              <Card key={policy.insurance_id}>
                <CardHeader>
                  <CardTitle>
                    {t('policyNumber')}: {policy.policy_number}
                  </CardTitle>
                  <CardDescription>Shipment: {policy.shipment_id}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <p className="text-sm">
                      <span className="font-medium">{t('coverageType')}:</span> {policy.coverage_type}
                    </p>
                    <p className="text-sm">
                      <span className="font-medium">{t('coverageAmount')}:</span>{' '}
                      {formatCurrency(policy.coverage_amount_usd)}
                    </p>
                    <p className="text-sm">
                      <span className="font-medium">{t('premium')}:</span> {formatCurrency(policy.premium_usd)}
                    </p>
                    <p className="text-sm">
                      <span className="font-medium">Valid Until:</span> {formatDate(policy.valid_until)}
                    </p>
                    <Link
                      href={`/${locale}/insurance/${policy.insurance_id}`}
                      className="mt-4 block"
                      aria-label={`View details for insurance policy ${policy.insurance_id}`}
                    >
                      <Button variant="outline" className="w-full">
                        View Details
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

export default InsurancePage;
