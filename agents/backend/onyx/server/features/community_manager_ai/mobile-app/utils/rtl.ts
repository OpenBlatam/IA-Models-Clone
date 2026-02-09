import { I18nManager } from 'react-native';
import * as Localization from 'expo-localization';

/**
 * RTL (Right-to-Left) support utilities
 */

const RTL_LANGUAGES = ['ar', 'he', 'fa', 'ur'];

/**
 * Check if current locale is RTL
 */
export function isRTL(): boolean {
  const locale = Localization.locale;
  return RTL_LANGUAGES.some((lang) => locale.startsWith(lang)) || I18nManager.isRTL;
}

/**
 * Force RTL layout
 */
export function forceRTL(rtl: boolean) {
  if (I18nManager.isRTL !== rtl) {
    I18nManager.forceRTL(rtl);
    // Note: App needs to be restarted for RTL changes to take effect
  }
}

/**
 * Get flex direction based on RTL
 */
export function getFlexDirection(): 'row' | 'row-reverse' {
  return isRTL() ? 'row-reverse' : 'row';
}

/**
 * Get text align based on RTL
 */
export function getTextAlign(): 'left' | 'right' {
  return isRTL() ? 'right' : 'left';
}

/**
 * Get start/end padding/margin
 */
export function getStartEnd(rtl?: boolean): { start: string; end: string } {
  const isRtlMode = rtl ?? isRTL();
  return {
    start: isRtlMode ? 'paddingRight' : 'paddingLeft',
    end: isRtlMode ? 'paddingLeft' : 'paddingRight',
  };
}


