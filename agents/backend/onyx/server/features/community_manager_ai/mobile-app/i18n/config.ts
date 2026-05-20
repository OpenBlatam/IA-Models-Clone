import * as Localization from 'expo-localization';
import { I18n } from 'i18n-js';

import en from './locales/en.json';
import es from './locales/es.json';

const i18n = new I18n({
  en,
  es,
});

// Set the locale once at the beginning of your app
i18n.locale = Localization.locale || 'en';

// When a value is missing from a language it'll fallback to another language with the key present
i18n.enableFallback = true;

// To see the fallback mechanism uncomment line below
// i18n.defaultLocale = 'en';

export default i18n;


