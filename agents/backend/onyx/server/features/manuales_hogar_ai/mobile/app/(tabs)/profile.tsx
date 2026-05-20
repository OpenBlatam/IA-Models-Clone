import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from '@/hooks/use-translation';
import { useApp } from '@/lib/context/app-context';
import { useAuth } from '@/lib/context/auth-context';
import { useSubscription } from '@/hooks/use-subscription';
import { manualService } from '@/services/api/manual-service';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';
import { SubscriptionStatus } from '@/components/subscription/subscription-status';
import i18n from '@/lib/i18n/config';

export default function ProfileScreen() {
  const { state, setTheme } = useApp();
  const { user, signOut } = useAuth();
  const router = useRouter();
  const { t } = useTranslation();
  const colors = state.colors;

  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['statistics'],
    queryFn: () => manualService.getStatistics({ days: 30 }),
  });

  const { subscription } = useSubscription();

  function changeLanguage(lang: 'en' | 'es') {
    i18n.changeLanguage(lang);
  }

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <LoadingSpinner />
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <ErrorMessage message="Error al cargar estadísticas" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          {user?.picture ? (
            <Image source={{ uri: user.picture }} style={styles.avatarImage} />
          ) : (
            <View style={[styles.avatar, { backgroundColor: colors.tint }]}>
              <Ionicons name="person" size={32} color="#FFFFFF" />
            </View>
          )}
          <Text style={[styles.name, { color: colors.text }]}>{user?.name || t('profile.title')}</Text>
          <Text style={[styles.email, { color: colors.textSecondary }]}>{user?.email}</Text>
        </View>

        <SubscriptionStatus />

        {stats && (
          <View style={[styles.statsCard, { backgroundColor: colors.card }]}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Estadísticas</Text>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={[styles.statValue, { color: colors.tint }]}>
                  {stats.total_manuals || 0}
                </Text>
                <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Manuales</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={[styles.statValue, { color: colors.tint }]}>
                  {stats.total_tokens || 0}
                </Text>
                <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Tokens</Text>
              </View>
            </View>
          </View>
        )}

        <View style={[styles.settingsCard, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('profile.settings')}</Text>
          
          <TouchableOpacity
            style={styles.settingItem}
            onPress={() => setTheme(state.theme === 'dark' ? 'light' : 'dark')}
          >
            <Ionicons name="moon-outline" size={24} color={colors.text} />
            <Text style={[styles.settingLabel, { color: colors.text }]}>{t('profile.darkMode')}</Text>
            <Ionicons
              name={state.isDark ? 'checkmark-circle' : 'ellipse-outline'}
              size={24}
              color={state.isDark ? colors.tint : colors.textSecondary}
            />
          </TouchableOpacity>

          <View style={styles.settingItem}>
            <Ionicons name="language-outline" size={24} color={colors.text} />
            <Text style={[styles.settingLabel, { color: colors.text }]}>{t('profile.language')}</Text>
            <View style={styles.languageButtons}>
              <TouchableOpacity
                style={[
                  styles.languageButton,
                  i18n.language === 'en' && { backgroundColor: colors.tint },
                ]}
                onPress={() => changeLanguage('en')}
              >
                <Text
                  style={[
                    styles.languageButtonText,
                    { color: i18n.language === 'en' ? '#FFFFFF' : colors.text },
                  ]}
                >
                  EN
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.languageButton,
                  i18n.language === 'es' && { backgroundColor: colors.tint },
                ]}
                onPress={() => changeLanguage('es')}
              >
                <Text
                  style={[
                    styles.languageButtonText,
                    { color: i18n.language === 'es' ? '#FFFFFF' : colors.text },
                  ]}
                >
                  ES
                </Text>
              </TouchableOpacity>
            </View>
          </View>


          <TouchableOpacity style={styles.settingItem} onPress={signOut}>
            <Ionicons name="log-out-outline" size={24} color={colors.error} />
            <Text style={[styles.settingLabel, { color: colors.error }]}>{t('auth.signOut')}</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  avatarImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginBottom: 12,
  },
  name: {
    fontSize: 24,
    fontWeight: '600',
  },
  email: {
    fontSize: 14,
    marginTop: 4,
  },
  statsCard: {
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 14,
  },
  settingsCard: {
    padding: 20,
    borderRadius: 12,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 12,
  },
  settingLabel: {
    flex: 1,
    fontSize: 16,
  },
  subscriptionCard: {
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  subscriptionInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  subscriptionPlan: {
    fontSize: 18,
    fontWeight: '600',
  },
  subscriptionStatus: {
    fontSize: 14,
    fontWeight: '500',
  },
  manageButton: {
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  manageButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  languageButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  languageButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  languageButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  upgradeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 8,
    gap: 8,
    marginTop: 12,
  },
  upgradeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});

