import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../types';

interface MenuItem {
  id: number;
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  showSwitch?: boolean;
  switchValue?: boolean;
  onSwitchChange?: (value: boolean) => void;
}

const ProfileScreen: React.FC = () => {
  const { userData, userId } = useSelector((state: RootState) => state.user);
  const dispatch = useDispatch();

  const [notificationsEnabled, setNotificationsEnabled] = React.useState(true);

  const menuItems: MenuItem[] = [
    {
      id: 1,
      title: 'Configuración',
      icon: 'settings-outline',
      onPress: () => {},
    },
    {
      id: 2,
      title: 'Notificaciones',
      icon: 'notifications-outline',
      onPress: () => {},
      showSwitch: true,
      switchValue: notificationsEnabled,
      onSwitchChange: setNotificationsEnabled,
    },
    {
      id: 3,
      title: 'Privacidad',
      icon: 'lock-closed-outline',
      onPress: () => {},
    },
    {
      id: 4,
      title: 'Ayuda y Soporte',
      icon: 'help-circle-outline',
      onPress: () => {},
    },
    {
      id: 5,
      title: 'Acerca de',
      icon: 'information-circle-outline',
      onPress: () => {},
    },
  ];

  const handleLogout = () => {
    dispatch({ type: 'LOGOUT' });
  };

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#6366f1', '#8b5cf6']}
        style={styles.header}
      >
        <View style={styles.avatarContainer}>
          <Ionicons name="person" size={48} color="#6366f1" />
        </View>
        <Text style={styles.userName}>
          {userData?.name || 'Usuario'}
        </Text>
        <Text style={styles.userEmail}>
          {userData?.email || userId || 'usuario@ejemplo.com'}
        </Text>
        {userData?.skin_type && (
          <View style={styles.skinTypeBadge}>
            <Ionicons name="body" size={16} color="#fff" />
            <Text style={styles.skinTypeText}>
              Tipo de piel: {userData.skin_type}
            </Text>
          </View>
        )}
      </LinearGradient>

      <View style={styles.content}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Cuenta</Text>
          {menuItems.map((item) => (
            <TouchableOpacity
              key={item.id}
              style={styles.menuItem}
              onPress={item.onPress}
              activeOpacity={0.7}
            >
              <View style={styles.menuItemLeft}>
                <Ionicons name={item.icon} size={24} color="#6366f1" />
                <Text style={styles.menuItemText}>{item.title}</Text>
              </View>
              {item.showSwitch ? (
                <Switch
                  value={item.switchValue}
                  onValueChange={item.onSwitchChange}
                  trackColor={{ false: '#d1d5db', true: '#6366f1' }}
                  thumbColor="#fff"
                />
              ) : (
                <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
              )}
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.statsSection}>
          <Text style={styles.sectionTitle}>Estadísticas</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Ionicons name="analytics" size={32} color="#6366f1" />
              <Text style={styles.statValue}>0</Text>
              <Text style={styles.statLabel}>Análisis</Text>
            </View>
            <View style={styles.statCard}>
              <Ionicons name="trophy" size={32} color="#f59e0b" />
              <Text style={styles.statValue}>0</Text>
              <Text style={styles.statLabel}>Logros</Text>
            </View>
            <View style={styles.statCard}>
              <Ionicons name="calendar" size={32} color="#10b981" />
              <Text style={styles.statValue}>0</Text>
              <Text style={styles.statLabel}>Días activo</Text>
            </View>
          </View>
        </View>

        <TouchableOpacity
          style={styles.logoutButton}
          onPress={handleLogout}
          activeOpacity={0.8}
        >
          <Ionicons name="log-out-outline" size={24} color="#ef4444" />
          <Text style={styles.logoutButtonText}>Cerrar Sesión</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 30,
    paddingTop: 60,
    alignItems: 'center',
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  avatarContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 8,
  },
  userName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 12,
  },
  skinTypeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginTop: 8,
  },
  skinTypeText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 6,
    textTransform: 'capitalize',
  },
  content: {
    padding: 20,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  menuItemText: {
    flex: 1,
    fontSize: 16,
    color: '#1f2937',
    marginLeft: 12,
  },
  statsSection: {
    marginBottom: 24,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginHorizontal: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 8,
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#ef4444',
    marginTop: 8,
  },
  logoutButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ef4444',
    marginLeft: 8,
  },
});

export default ProfileScreen;

