import { create } from 'zustand';
import { Language } from '../i18n/translations';

interface I18nState {
  language: Language;
  setLanguage: (lang: Language) => void;
}

// Load from localStorage
const getStoredLanguage = (): Language => {
  if (typeof window === 'undefined') return 'es';
  const stored = localStorage.getItem('robot-language');
  return (stored as Language) || 'es';
};

const setStoredLanguage = (lang: Language) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('robot-language', lang);
    document.documentElement.lang = lang;
  }
};

export const useI18nStore = create<I18nState>((set) => ({
  language: getStoredLanguage(),
  setLanguage: (lang) => {
    setStoredLanguage(lang);
    set({ language: lang });
  },
}));

