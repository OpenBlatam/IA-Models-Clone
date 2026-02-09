/**
 * Language Selector Component
 * @module robot-3d-view/controls/language-selector
 */

'use client';

import { memo } from 'react';
import { useI18n } from '../hooks/use-i18n';

/**
 * Language Selector Component
 * 
 * Provides UI for selecting language.
 * 
 * @returns Language selector component
 */
export const LanguageSelector = memo(() => {
  const { t, setLanguage, currentLanguage, getAvailableLanguages } = useI18n();
  const languages = getAvailableLanguages();

  return (
    <div className="absolute top-4 left-4 z-40">
      <div className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-2 shadow-lg">
        <div className="text-[10px] text-gray-400 mb-2 px-2">{t('common.settings')}:</div>
        <select
          value={currentLanguage}
          onChange={(e) => setLanguage(e.target.value)}
          className="px-2 py-1 text-[10px] rounded bg-gray-700/50 hover:bg-gray-600 border border-gray-600 text-white transition-all"
          aria-label="Select language"
        >
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.nativeName}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
});

LanguageSelector.displayName = 'LanguageSelector';



