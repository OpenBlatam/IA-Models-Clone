/**
 * Alert Component
 * ==============
 * Custom alert component
 */

import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { CustomModal } from './modal';

interface AlertProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  buttons?: Array<{
    text: string;
    onPress: () => void;
    style?: 'default' | 'cancel' | 'destructive';
  }>;
}

export function Alert({
  visible,
  onClose,
  title,
  message,
  type = 'info',
  buttons,
}: AlertProps) {
  const { state } = useApp();
  const colors = state.colors;

  const typeConfig = {
    success: { icon: 'checkmark-circle', color: colors.success },
    error: { icon: 'close-circle', color: colors.error },
    warning: { icon: 'warning', color: colors.warning },
    info: { icon: 'information-circle', color: colors.info },
  };

  const config = typeConfig[type];
  const defaultButtons = buttons || [
    {
      text: 'OK',
      onPress: onClose,
      style: 'default' as const,
    },
  ];

  return (
    <CustomModal visible={visible} onClose={onClose} title={title} showCloseButton={false}>
      <View style={styles.content}>
        <View style={[styles.iconContainer, { backgroundColor: `${config.color}20` }]}>
          <Ionicons name={config.icon as any} size={48} color={config.color} />
        </View>
        <Text style={[styles.message, { color: colors.text }]}>{message}</Text>
        <View style={styles.buttons}>
          {defaultButtons.map((button, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.button,
                {
                  backgroundColor:
                    button.style === 'destructive'
                      ? colors.error
                      : button.style === 'cancel'
                      ? colors.backgroundSecondary
                      : colors.tint,
                },
                index > 0 && styles.buttonMargin,
              ]}
              onPress={button.onPress}
            >
              <Text
                style={[
                  styles.buttonText,
                  {
                    color:
                      button.style === 'cancel' ? colors.text : '#FFFFFF',
                  },
                ]}
              >
                {button.text}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </CustomModal>
  );
}

const styles = StyleSheet.create({
  content: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  message: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 22,
  },
  buttons: {
    flexDirection: 'row',
    width: '100%',
    gap: 12,
  },
  button: {
    flex: 1,
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonMargin: {
    marginLeft: 0,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
  },
});



