import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  TouchableWithoutFeedback,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface Action {
  label: string;
  icon?: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  destructive?: boolean;
}

interface ActionSheetProps {
  visible: boolean;
  onClose: () => void;
  actions: Action[];
  title?: string;
  cancelText?: string;
}

const ActionSheet: React.FC<ActionSheetProps> = ({
  visible,
  onClose,
  actions,
  title,
  cancelText = 'Cancelar',
}) => {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();

  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="slide"
      onRequestClose={onClose}
    >
      <TouchableWithoutFeedback onPress={onClose}>
        <View style={styles.overlay}>
          <TouchableWithoutFeedback>
            <View
              style={[
                styles.container,
                {
                  backgroundColor: colors.card,
                  paddingBottom: insets.bottom,
                },
              ]}
            >
              {title && (
                <View style={styles.header}>
                  <Text style={[styles.title, { color: colors.text }]}>
                    {title}
                  </Text>
                </View>
              )}
              {actions.map((action, index) => (
                <TouchableOpacity
                  key={index}
                  style={[
                    styles.action,
                    {
                      borderBottomColor: colors.border,
                      borderBottomWidth: index < actions.length - 1 ? 1 : 0,
                    },
                  ]}
                  onPress={() => {
                    action.onPress();
                    onClose();
                  }}
                  activeOpacity={0.7}
                >
                  {action.icon && (
                    <Ionicons
                      name={action.icon}
                      size={20}
                      color={action.destructive ? colors.error : colors.text}
                      style={styles.icon}
                    />
                  )}
                  <Text
                    style={[
                      styles.actionText,
                      {
                        color: action.destructive ? colors.error : colors.text,
                      },
                    ]}
                  >
                    {action.label}
                  </Text>
                </TouchableOpacity>
              ))}
              <TouchableOpacity
                style={[
                  styles.cancel,
                  {
                    backgroundColor: colors.surface,
                    marginTop: 8,
                  },
                ]}
                onPress={onClose}
                activeOpacity={0.7}
              >
                <Text style={[styles.cancelText, { color: colors.text }]}>
                  {cancelText}
                </Text>
              </TouchableOpacity>
            </View>
          </TouchableWithoutFeedback>
        </View>
      </TouchableWithoutFeedback>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  container: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingTop: 16,
  },
  header: {
    paddingHorizontal: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  action: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  icon: {
    marginRight: 12,
  },
  actionText: {
    fontSize: 16,
    flex: 1,
  },
  cancel: {
    padding: 16,
    alignItems: 'center',
    borderRadius: 12,
    marginHorizontal: 16,
    marginBottom: 16,
  },
  cancelText: {
    fontSize: 16,
    fontWeight: '600',
  },
});

export default ActionSheet;

