'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Navbar from '@/components/layout/navbar';
import { reportsApi } from '@/lib/api/reports';
import { getErrorMessage } from '@/lib/error-handler';
import Loading from '@/components/ui/loading';
import ErrorMessage from '@/components/ui/error-message';

const ReportsPage = () => {
  const t = useTranslations();

  const { data: dashboardData, isLoading, error } = useQuery({
    queryKey: ['reports', 'dashboard'],
    queryFn: () => reportsApi.getDashboard(),
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('nav.reports')}</h1>
            <p className="text-muted-foreground">View reports and analytics</p>
          </div>
          <Loading text="Loading reports..." />
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{t('nav.reports')}</h1>
            <p className="text-muted-foreground">View reports and analytics</p>
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
        <div className="mb-8">
          <h1 className="text-3xl font-bold">{t('nav.reports')}</h1>
          <p className="text-muted-foreground">View reports and analytics</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {dashboardData && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Dashboard Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {JSON.stringify(dashboardData, null, 2)}
                  </p>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default ReportsPage;




