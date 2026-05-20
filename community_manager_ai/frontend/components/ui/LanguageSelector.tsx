'use client';

import { useLocale } from 'next-intl';
import { useRouter, usePathname } from '@/i18n/routing';
import { locales, localeNames, localeFlags } from '@/i18n/config';
import { Dropdown } from './Dropdown';
import { Globe } from 'lucide-react';

export const LanguageSelector = () => {
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  const handleLanguageChange = (newLocale: string) => {
    router.replace(pathname, { locale: newLocale });
  };

  const options = locales.map((loc) => ({
    label: `${localeFlags[loc]} ${localeNames[loc]}`,
    value: loc,
  }));

  const currentOption = options.find((opt) => opt.value === locale) || options[0];

  return (
    <div className="flex items-center gap-2">
      <Globe className="h-4 w-4 text-gray-500" />
      <Dropdown
        options={options}
        value={locale}
        onSelect={handleLanguageChange}
        className="w-40"
      />
    </div>
  );
};



