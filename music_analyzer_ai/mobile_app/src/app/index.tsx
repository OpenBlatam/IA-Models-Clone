import React, { useCallback } from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../constants/config';
import { useHealthCheck } from '../hooks/use-music-analysis';
import { useTranslation } from '../hooks/use-translation';
import { useMusic } from '../stores/music';
import { Button } from '../components/common/button';
import { AnimatedView } from '../components/common/animated-view';

export default function HomeScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const { data: health } = useHealthCheck();
  const { favorites } = useMusic();

  const handleSearchPress = useCallback(() => {
    router.push({
      pathname: '/search',
      params: {},
    });
  }, [router]);

  const handleFavoritesPress = useCallback(() => {
    router.push({
      pathname: '/favorites',
      params: {},
    });
  }, [router]);

  const handleComparePress = useCallback(() => {
    router.push({
      pathname: '/compare',
      params: {},
    });
  }, [router]);

  const handleHistoryPress = useCallback(() => {
    router.push({
      pathname: '/history',
      params: {},
    });
  }, [router]);

  const handleDiscoveryPress = useCallback(() => {
    router.push({
      pathname: '/discovery',
      params: {},
    });
  }, [router]);

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Music Analyzer AI</Text>
          <Text style={styles.subtitle}>
            Analyze music with AI-powered insights
          </Text>
        </View>

        <View style={styles.statusContainer}>
          <View style={styles.statusRow}>
            <View
              style={[
                styles.statusIndicator,
                health?.status === 'healthy'
                  ? styles.statusHealthy
                  : styles.statusUnhealthy,
              ]}
            />
            <Text style={styles.statusText}>
              {health?.status === 'healthy' ? 'Connected' : 'Disconnected'}
            </Text>
          </View>
        </View>

        <AnimatedView delay={100}>
          <Button
            title={`${t('common.search')} & Analyze`}
            onPress={handleSearchPress}
            variant="primary"
            size="large"
            fullWidth
            style={styles.searchButton}
          />
        </AnimatedView>

        <View style={styles.actionsContainer}>
          {favorites.length > 0 && (
            <AnimatedView delay={200}>
              <Button
                title={`${t('common.favorites')} (${favorites.length})`}
                onPress={handleFavoritesPress}
                variant="secondary"
                fullWidth
                style={styles.actionButton}
              />
            </AnimatedView>
          )}
          <AnimatedView delay={300}>
            <Button
              title="Compare Tracks"
              onPress={handleComparePress}
              variant="secondary"
              fullWidth
              style={styles.actionButton}
            />
          </AnimatedView>
          <AnimatedView delay={400}>
            <Button
              title={t('common.history')}
              onPress={handleHistoryPress}
              variant="secondary"
              fullWidth
              style={styles.actionButton}
            />
          </AnimatedView>
          <AnimatedView delay={500}>
            <Button
              title="🔍 Discover"
              onPress={handleDiscoveryPress}
              variant="secondary"
              fullWidth
              style={styles.actionButton}
            />
          </AnimatedView>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    flex: 1,
    padding: SPACING.lg,
    justifyContent: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: SPACING.xxl,
  },
  title: {
    ...TYPOGRAPHY.h1,
    color: COLORS.text,
    marginBottom: SPACING.sm,
    textAlign: 'center',
  },
  subtitle: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
  statusContainer: {
    marginBottom: SPACING.xl,
    alignItems: 'center',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: SPACING.sm,
  },
  statusHealthy: {
    backgroundColor: COLORS.success,
  },
  statusUnhealthy: {
    backgroundColor: COLORS.error,
  },
  statusText: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
  },
  searchButton: {
    marginBottom: SPACING.md,
  },
  actionsContainer: {
    width: '100%',
    gap: SPACING.sm,
    marginTop: SPACING.md,
  },
  actionButton: {
    marginBottom: SPACING.xs,
  },
});

