import { ReactNode } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Toast from 'react-native-toast-message';
import { useTheme } from '@/contexts/ThemeContext';

interface ToastProviderProps {
  children: ReactNode;
}

export function ToastProvider({ children }: ToastProviderProps) {
  const { colors } = useTheme();

  return (
    <>
      {children}
      <Toast
        config={{
          success: (props) => (
            <View style={[styles.toast, styles.success, { backgroundColor: colors.success }]}>
              <Ionicons name="checkmark-circle" size={24} color="#fff" />
              <Text style={styles.text}>{props.text1}</Text>
            </View>
          ),
          error: (props) => (
            <View style={[styles.toast, styles.error, { backgroundColor: colors.error }]}>
              <Ionicons name="close-circle" size={24} color="#fff" />
              <Text style={styles.text}>{props.text1}</Text>
            </View>
          ),
          info: (props) => (
            <View style={[styles.toast, styles.info, { backgroundColor: colors.primary }]}>
              <Ionicons name="information-circle" size={24} color="#fff" />
              <Text style={styles.text}>{props.text1}</Text>
            </View>
          ),
        }}
      />
    </>
  );
}

const styles = StyleSheet.create({
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 8,
    marginHorizontal: 16,
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  success: {},
  error: {},
  info: {},
  text: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
});
