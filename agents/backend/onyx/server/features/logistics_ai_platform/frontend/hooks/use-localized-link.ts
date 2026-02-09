import { useLocale } from './use-locale';

export const useLocalizedLink = () => {
  const locale = useLocale();

  const getLink = (path: string): string => {
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `/${locale}${cleanPath}`;
  };

  return { getLink, locale };
};




