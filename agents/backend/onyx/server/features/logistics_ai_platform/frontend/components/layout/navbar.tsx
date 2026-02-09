'use client';

import { signOut } from 'next-auth/react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const Navbar = () => {
  const t = useTranslations();
  const pathname = usePathname();

  const locale = pathname?.split('/')[1] || 'en';
  
  const navItems = [
    { href: `/${locale}/dashboard`, label: t('nav.dashboard') },
    { href: `/${locale}/quotes`, label: t('nav.quotes') },
    { href: `/${locale}/bookings`, label: t('nav.bookings') },
    { href: `/${locale}/shipments`, label: t('nav.shipments') },
    { href: `/${locale}/tracking`, label: t('nav.tracking') },
    { href: `/${locale}/containers`, label: t('nav.containers') },
    { href: `/${locale}/invoices`, label: t('nav.invoices') },
    { href: `/${locale}/insurance`, label: t('nav.insurance') },
    { href: `/${locale}/documents`, label: t('nav.documents') },
    { href: `/${locale}/alerts`, label: t('nav.alerts') },
    { href: `/${locale}/reports`, label: t('nav.reports') },
  ];

  const handleSignOut = async () => {
    const locale = pathname?.split('/')[1] || 'en';
    await signOut({ callbackUrl: `/${locale}/auth/signin` });
  };

  return (
    <nav className="border-b bg-background" role="navigation" aria-label="Main navigation">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center space-x-4 md:space-x-8">
          <Link
            href={`/${locale}/dashboard`}
            className="text-xl font-bold hover:text-primary transition-colors"
            aria-label="Logistics AI Home"
          >
            Logistics AI
          </Link>
          <div className="hidden md:flex md:space-x-4">
            {navItems.map((item) => {
              const isActive = pathname?.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'text-sm font-medium transition-colors hover:text-primary',
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  )}
                  aria-label={item.label}
                  aria-current={isActive ? 'page' : undefined}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
        <Button
          variant="ghost"
          onClick={handleSignOut}
          aria-label={t('auth.signOut')}
          className="whitespace-nowrap"
        >
          {t('auth.signOut')}
        </Button>
      </div>
    </nav>
  );
};

export default Navbar;
