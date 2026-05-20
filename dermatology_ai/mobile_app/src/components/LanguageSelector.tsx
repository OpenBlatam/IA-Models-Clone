import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useLocalization } from '../hooks/useLocalization';

const LanguageSelector: React.FC = () => {
  const { language, changeLanguage, t } = useLocalization();

  const languages = [
    { code: 'es' as const, label: 'Español', flag: '🇪🇸' },
    { code: 'en' as const, label: 'English', flag: '🇺🇸' },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.label}>{t('common.language') || 'Idioma'}</Text>
      <View style={styles.languages}>
        {languages.map((lang) => (
          <TouchableOpacity
            key={lang.code}
            style={[
              styles.languageButton,
              language === lang.code && styles.languageButtonActive,
            ]}
            onPress={() => changeLanguage(lang.code)}
            activeOpacity={0.7}
          >
            <Text style={styles.flag}>{lang.flag}</Text>
            <Text
              style={[
                styles.languageText,
                language === lang.code && styles.languageTextActive,
              ]}
            >
              {lang.label}
            </Text>
            {language === lang.code && (
              <Ionicons name="checkmark" size={20} color="#6366f1" />
            )}
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 24,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  languages: {
    gap: 8,
  },
  languageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  languageButtonActive: {
    borderColor: '#6366f1',
    backgroundColor: '#f3f4f6',
  },
  flag: {
    fontSize: 24,
    marginRight: 12,
  },
  languageText: {
    flex: 1,
    fontSize: 16,
    color: '#1f2937',
  },
  languageTextActive: {
    fontWeight: '600',
    color: '#6366f1',
  },
});

export default LanguageSelector;

