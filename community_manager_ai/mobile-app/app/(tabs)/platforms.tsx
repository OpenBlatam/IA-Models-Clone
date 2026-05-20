import { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  TextInput,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { usePlatforms, useConnectPlatform, useDisconnectPlatform } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { PLATFORMS } from '@/utils/constants';

export default function PlatformsScreen() {
  const { data: connectedPlatforms, isLoading, refetch } = usePlatforms();
  const connectPlatform = useConnectPlatform();
  const disconnectPlatform = useDisconnectPlatform();
  const [connectingPlatform, setConnectingPlatform] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<Record<string, string>>({});

  const isConnected = (platformId: string) => {
    return connectedPlatforms?.includes(platformId) || false;
  };

  const handleConnect = (platformId: string) => {
    setConnectingPlatform(platformId);
    Alert.prompt(
      `Connect ${PLATFORMS.find((p) => p.id === platformId)?.name}`,
      'Enter your API token or access key:',
      [
        { text: 'Cancel', style: 'cancel', onPress: () => setConnectingPlatform(null) },
        {
          text: 'Connect',
          onPress: async (token) => {
            if (!token) {
              Alert.alert('Error', 'Token is required');
              setConnectingPlatform(null);
              return;
            }

            try {
              await connectPlatform.mutateAsync({
                platform: platformId,
                credentials: { token },
              });
              Alert.alert('Success', `Connected to ${PLATFORMS.find((p) => p.id === platformId)?.name}`);
              refetch();
            } catch (error) {
              Alert.alert('Error', 'Failed to connect platform');
            } finally {
              setConnectingPlatform(null);
            }
          },
        },
      ],
      'plain-text'
    );
  };

  const handleDisconnect = (platformId: string) => {
    Alert.alert(
      'Disconnect Platform',
      `Are you sure you want to disconnect ${PLATFORMS.find((p) => p.id === platformId)?.name}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Disconnect',
          style: 'destructive',
          onPress: async () => {
            try {
              await disconnectPlatform.mutateAsync(platformId);
              Alert.alert('Success', 'Platform disconnected');
              refetch();
            } catch (error) {
              Alert.alert('Error', 'Failed to disconnect platform');
            }
          },
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
      >
        <View style={styles.content}>
          <Text style={styles.title}>Social Platforms</Text>
          <Text style={styles.subtitle}>
            Connect your social media accounts to start publishing
          </Text>

          {isLoading && !connectedPlatforms ? (
            <ActivityIndicator size="large" color="#0ea5e9" style={styles.loader} />
          ) : (
            <View style={styles.platformsList}>
              {PLATFORMS.map((platform) => {
                const connected = isConnected(platform.id);
                const connecting = connectingPlatform === platform.id;

                return (
                  <View key={platform.id} style={styles.platformCard}>
                    <View style={styles.platformHeader}>
                      <View style={[styles.platformIcon, { backgroundColor: platform.color + '20' }]}>
                        <Ionicons name={platform.icon as any} size={32} color={platform.color} />
                      </View>
                      <View style={styles.platformInfo}>
                        <Text style={styles.platformName}>{platform.name}</Text>
                        <View style={styles.statusContainer}>
                          <View
                            style={[
                              styles.statusDot,
                              { backgroundColor: connected ? '#10b981' : '#9ca3af' },
                            ]}
                          />
                          <Text style={styles.statusText}>
                            {connected ? 'Connected' : 'Not Connected'}
                          </Text>
                        </View>
                      </View>
                    </View>

                    <TouchableOpacity
                      style={[
                        styles.actionButton,
                        connected ? styles.disconnectButton : styles.connectButton,
                        connecting && styles.buttonDisabled,
                      ]}
                      onPress={() => (connected ? handleDisconnect(platform.id) : handleConnect(platform.id))}
                      disabled={connecting}
                    >
                      {connecting ? (
                        <ActivityIndicator size="small" color="#fff" />
                      ) : (
                        <Text style={styles.actionButtonText}>
                          {connected ? 'Disconnect' : 'Connect'}
                        </Text>
                      )}
                    </TouchableOpacity>
                  </View>
                );
              })}
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 24,
  },
  loader: {
    marginTop: 40,
  },
  platformsList: {
    gap: 16,
  },
  platformCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  platformHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  platformIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  platformInfo: {
    flex: 1,
  },
  platformName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 12,
    color: '#6b7280',
  },
  actionButton: {
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  connectButton: {
    backgroundColor: '#0ea5e9',
  },
  disconnectButton: {
    backgroundColor: '#ef4444',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});

