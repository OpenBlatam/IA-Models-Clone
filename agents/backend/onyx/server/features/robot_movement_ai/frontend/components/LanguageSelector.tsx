'use client';

import { Globe } from 'lucide-react';
import { useI18nStore } from '@/lib/store/i18nStore';
import { Language } from '@/lib/i18n/translations';

export default function LanguageSelector() {
  const { language, setLanguage } = useI18nStore();

  const languages: { code: Language; name: string; flag: string }[] = [
    { code: 'es', name: 'Español', flag: '🇪🇸' },
    { code: 'en', name: 'English', flag: '🇺🇸' },
  ];

  return (
    <div className="relative group">
      <button className="flex items-center gap-tesla-sm px-tesla-sm py-tesla-sm bg-white border border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium">
        <Globe className="w-4 h-4 text-tesla-gray-dark" />
        <span className="hidden sm:inline">
          {languages.find((l) => l.code === language)?.flag} {languages.find((l) => l.code === language)?.name}
        </span>
      </button>
      <div className="absolute right-0 mt-tesla-sm w-48 bg-white border border-gray-200 rounded-md shadow-tesla-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
        {languages.map((lang) => (
          <button
            key={lang.code}
            onClick={() => setLanguage(lang.code)}
            className={`w-full flex items-center gap-tesla-sm px-tesla-md py-tesla-sm text-left hover:bg-gray-50 transition-colors rounded-md ${
              language === lang.code ? 'bg-gray-50' : ''
            }`}
          >
            <span className="text-2xl">{lang.flag}</span>
            <span className="text-tesla-black font-medium">{lang.name}</span>
            {language === lang.code && (
              <span className="ml-auto text-tesla-blue font-semibold">✓</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}


