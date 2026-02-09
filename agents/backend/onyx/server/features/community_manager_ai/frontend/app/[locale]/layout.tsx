import { ReactNode } from 'react';
import { NextIntlClientProvider } from 'next-intl';
import { getMessages } from 'next-intl/server';
import { Providers } from '../providers';
import { ThemeInitializer } from '@/components/theme/ThemeInitializer';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { ScrollToTop } from '@/components/ui/ScrollToTop';
import { KeyboardShortcuts } from '@/components/ui/KeyboardShortcuts';

export default async function LocaleLayout({
  children,
  params,
}: {
  children: ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const messages = await getMessages();

  return (
    <html lang={locale} suppressHydrationWarning>
      <body>
        <ThemeInitializer />
        <ErrorBoundary>
          <NextIntlClientProvider messages={messages}>
            <Providers>
              <KeyboardShortcuts />
              {children}
              <ScrollToTop />
            </Providers>
          </NextIntlClientProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
