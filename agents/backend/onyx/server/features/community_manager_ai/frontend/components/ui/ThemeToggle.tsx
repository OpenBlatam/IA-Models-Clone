'use client';

import { Sun, Moon, Monitor } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import { useTranslations } from 'next-intl';
import { Button } from './Button';
import { Tooltip } from './Tooltip';
import { cn } from '@/lib/utils';

export const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();
  const t = useTranslations('theme');

  const getIcon = () => {
    if (theme === 'light') return Sun;
    if (theme === 'dark') return Moon;
    return Monitor;
  };

  const getLabel = () => {
    if (theme === 'light') return t('light');
    if (theme === 'dark') return t('dark');
    return t('system');
  };

  const Icon = getIcon();

  return (
    <Tooltip content={getLabel()}>
      <Button
        variant="ghost"
        size="sm"
        onClick={toggleTheme}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleTheme();
          }
        }}
        aria-label={t('toggleTheme')}
        tabIndex={0}
        className="relative"
      >
        <Icon className="h-5 w-5" />
        <span className="sr-only">{getLabel()}</span>
      </Button>
    </Tooltip>
  );
};

export const ThemeSelect = () => {
  const { theme, setTheme } = useTheme();
  const t = useTranslations('theme');

  const options = [
    { value: 'light' as const, label: t('light'), icon: Sun },
    { value: 'dark' as const, label: t('dark'), icon: Moon },
    { value: 'system' as const, label: t('system'), icon: Monitor },
  ];

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        {t('theme')}
      </label>
      <div className="flex gap-2">
        {options.map((option) => {
          const Icon = option.icon;
          const isActive = theme === option.value;

          return (
            <button
              key={option.value}
              type="button"
              onClick={() => setTheme(option.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setTheme(option.value);
                }
              }}
              className={cn(
                'flex flex-col items-center gap-2 rounded-lg border-2 p-3 transition-all',
                'hover:bg-gray-100 dark:hover:bg-gray-800',
                'focus:outline-none focus:ring-2 focus:ring-primary-500',
                isActive
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-200 dark:border-gray-700'
              )}
              aria-label={option.label}
              aria-pressed={isActive}
              tabIndex={0}
            >
              <Icon className="h-5 w-5" />
              <span className="text-xs font-medium">{option.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};



