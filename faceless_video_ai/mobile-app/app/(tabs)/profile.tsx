import React from 'react';
import { View, Text, StyleSheet, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAuthStore } from '@/store/auth-store';
import { Button } from '@/components/ui/button';
import { useQuota } from '@/hooks/use-analytics';

export default function ProfileScreen() {
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const { data: quota } = useQuota();

  const handleLogout = async () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        style: 'destructive',
        onPress: async () => {
          await logout();
          router.replace('/(auth)/login');
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
        <Text style={styles.title}>Profile</Text>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Information</Text>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Email:</Text>
            <Text style={styles.infoValue}>{user?.email}</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>User ID:</Text>
            <Text style={styles.infoValue}>{user?.user_id}</Text>
          </View>
          {user?.roles && user.roles.length > 0 && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Roles:</Text>
              <Text style={styles.infoValue}>{user.roles.join(', ')}</Text>
            </View>
          )}
        </View>

        {quota && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Quota</Text>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Videos Generated:</Text>
              <Text style={styles.infoValue}>
                {quota.videos_generated} / {quota.videos_limit}
              </Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Storage Used:</Text>
              <Text style={styles.infoValue}>
                {(quota.storage_used / 1024 / 1024).toFixed(2)} MB /{' '}
                {(quota.storage_limit / 1024 / 1024).toFixed(2)} MB
              </Text>
            </View>
          </View>
        )}

        <View style={styles.actionsContainer}>
          <Button
            title="Settings"
            onPress={() => router.push('/settings')}
            variant="outline"
            size="large"
            style={styles.actionButton}
          />
          <Button
            title="Analytics"
            onPress={() => router.push('/analytics')}
            variant="outline"
            size="large"
            style={styles.actionButton}
          />
          <Button
            title="Logout"
            onPress={handleLogout}
            variant="danger"
            size="large"
            style={styles.actionButton}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 24,
    color: '#000000',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#000000',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  infoLabel: {
    fontSize: 14,
    color: '#666666',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000000',
  },
  actionsContainer: {
    marginTop: 24,
  },
  actionButton: {
    marginBottom: 12,
  },
});


