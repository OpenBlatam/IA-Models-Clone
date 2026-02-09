import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useLocalization } from '../../hooks/use-localization';

interface Language {
  code: string;
  name: string;
  nativeName: string;
}

interface LanguageSelectorProps {
  languages?: Language[];
  onLanguageChange?: (code: string) => void;
}

const defaultLanguages: Language[] = [
  { code: 'en', name: 'English', nativeName: 'English' },
  { code: 'es', name: 'Spanish', nativeName: 'Español' },
  { code: 'fr', name: 'French', nativeName: 'Français' },
  { code: 'de', name: 'German', nativeName: 'Deutsch' },
];

/**
 * Language selector component
 * Allows users to change app language
 */
export function LanguageSelector({
  languages = defaultLanguages,
  onLanguageChange,
}: LanguageSelectorProps) {
  const { language, changeLanguage } = useLocalization();

  const handleLanguageChange = (code: string) => {
    changeLanguage(code);
    onLanguageChange?.(code);
  };

  return (
    <View style={styles.container}>
      {languages.map((lang) => {
        const isSelected = language === lang.code;

        return (
          <TouchableOpacity
            key={lang.code}
            style={[
              styles.languageButton,
              isSelected && styles.selectedLanguageButton,
            ]}
            onPress={() => handleLanguageChange(lang.code)}
            accessibilityRole="button"
            accessibilityState={{ selected: isSelected }}
          >
            <Text
              style={[
                styles.languageText,
                isSelected && styles.selectedLanguageText,
              ]}
            >
              {lang.nativeName}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
  },
  languageButton: {
    paddingVertical: SPACING.sm,
    paddingHorizontal: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.surfaceLight,
  },
  selectedLanguageButton: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  languageText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
  },
  selectedLanguageText: {
    color: COLORS.surface,
    fontWeight: '600',
  },
});

