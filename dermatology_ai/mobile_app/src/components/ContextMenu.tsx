import React, { useState } from 'react';
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
  Modal,
  TouchableWithoutFeedback,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface MenuItem {
  label: string;
  onPress: () => void;
  icon?: React.ReactNode;
  destructive?: boolean;
  disabled?: boolean;
}

interface ContextMenuProps {
  trigger: React.ReactNode;
  items: MenuItem[];
  position?: 'top' | 'bottom' | 'left' | 'right';
}

const ContextMenu: React.FC<ContextMenuProps> = ({
  trigger,
  items,
  position = 'bottom',
}) => {
  const { colors } = useTheme();
  const [visible, setVisible] = useState(false);
  const [triggerLayout, setTriggerLayout] = useState({ x: 0, y: 0, width: 0, height: 0 });

  const handlePress = (event: any) => {
    const { pageX, pageY } = event.nativeEvent;
    setTriggerLayout({
      x: pageX,
      y: pageY,
      width: 0,
      height: 0,
    });
    setVisible(true);
  };

  const getMenuPosition = () => {
    switch (position) {
      case 'top':
        return { bottom: triggerLayout.y + triggerLayout.height };
      case 'bottom':
        return { top: triggerLayout.y };
      case 'left':
        return { right: triggerLayout.x };
      case 'right':
        return { left: triggerLayout.x };
      default:
        return { top: triggerLayout.y };
    }
  };

  return (
    <>
      <TouchableOpacity onPress={handlePress} activeOpacity={0.7}>
        {trigger}
      </TouchableOpacity>

      <Modal
        visible={visible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setVisible(false)}
      >
        <TouchableWithoutFeedback onPress={() => setVisible(false)}>
          <View style={styles.overlay}>
            <TouchableWithoutFeedback>
              <View
                style={[
                  styles.menu,
                  {
                    backgroundColor: colors.card,
                    ...getMenuPosition(),
                  },
                ]}
              >
                {items.map((item, index) => (
                  <TouchableOpacity
                    key={index}
                    style={[
                      styles.menuItem,
                      {
                        borderBottomColor: colors.border,
                        borderBottomWidth: index < items.length - 1 ? 1 : 0,
                      },
                      item.disabled && styles.disabled,
                    ]}
                    onPress={() => {
                      if (!item.disabled) {
                        item.onPress();
                        setVisible(false);
                      }
                    }}
                    disabled={item.disabled}
                    activeOpacity={0.7}
                  >
                    {item.icon && <View style={styles.icon}>{item.icon}</View>}
                    <Text
                      style={[
                        styles.menuItemText,
                        {
                          color: item.destructive
                            ? colors.error
                            : item.disabled
                            ? colors.textSecondary
                            : colors.text,
                        },
                      ]}
                    >
                      {item.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
  },
  menu: {
    position: 'absolute',
    minWidth: 200,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 10,
    overflow: 'hidden',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
  },
  icon: {
    marginRight: 12,
  },
  menuItemText: {
    fontSize: 14,
    flex: 1,
  },
  disabled: {
    opacity: 0.5,
  },
});

export default ContextMenu;

