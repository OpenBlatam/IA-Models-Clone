/**
 * Login Screen
 * ============
 * Login screen with Google OAuth
 */

import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '@/lib/context/auth-context';
import { useApp } from '@/lib/context/app-context';
import { useTranslation } from '@/hooks/use-translation';
import { Ionicons } from '@expo/vector-icons';

export function LoginScreen() {
  const { signInWithGoogle, isLoading } = useAuth();
  const { state } = useApp();
  const { t } = useTranslation();
  const colors = state.colors;

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top', 'bottom']}>
      <View style={styles.content}>
        <View style={styles.header}>
          <Ionicons name="home" size={64} color={colors.tint} />
          <Text style={[styles.title, { color: colors.text }]}>{t('home.title')}</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
            {t('home.subtitle')}
          </Text>
        </View>

        <View style={styles.loginSection}>
          <TouchableOpacity
            style={[styles.googleButton, { backgroundColor: '#FFFFFF', borderColor: colors.border }]}
            onPress={signInWithGoogle}
            disabled={isLoading}
            activeOpacity={0.7}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color={colors.text} />
            ) : (
              <>
                <Ionicons name="logo-google" size={24} color="#4285F4" />
                <Text style={[styles.googleButtonText, { color: colors.text }]}>
                  {t('auth.signInWithGoogle')}
                </Text>
              </>
            )}
          </TouchableOpacity>

          <Text style={[styles.terms, { color: colors.textSecondary }]}>
            {t('auth.terms', 'By signing in, you agree to our Terms of Service and Privacy Policy')}
          </Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 48,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
  },
  loginSection: {
    width: '100%',
  },
  googleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    gap: 12,
    marginBottom: 16,
  },
  googleButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  terms: {
    fontSize: 12,
    textAlign: 'center',
    lineHeight: 18,
  },
});


