'use client';

import { ReactNode } from 'react';
import { PageLayout } from '@/components/layout/page-layout';
import { BackButton } from '@/components/ui/back-button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface FormLayoutProps {
  title: string;
  backHref: string;
  backLabel?: string;
  children: ReactNode;
}

const FormLayout = ({ title, backHref, backLabel, children }: FormLayoutProps) => {
  return (
    <PageLayout>
      <div className="max-w-3xl mx-auto">
        <BackButton href={backHref} label={backLabel} className="mb-6" />

        <Card>
          <CardHeader>
            <CardTitle>{title}</CardTitle>
          </CardHeader>
          <CardContent>{children}</CardContent>
        </Card>
      </div>
    </PageLayout>
  );
};

export { FormLayout };

