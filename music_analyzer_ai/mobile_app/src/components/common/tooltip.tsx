import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
} from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface TooltipProps {
  children: React.ReactNode;
  content: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
}

/**
 * Tooltip component
 * Contextual help text
 */
export function Tooltip({
  children,
  content,
  position = 'top',
  delay = 0,
}: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [layout, setLayout] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);

  const handlePress = () => {
    if (delay > 0) {
      setTimeout(() => setVisible(true), delay);
    } else {
      setVisible(true);
    }
  };

  const handleLayout = (event: {
    nativeEvent: { layout: { x: number; y: number; width: number; height: number } };
  }) => {
    setLayout(event.nativeEvent.layout);
  };

  return (
    <>
      <TouchableOpacity
        onPress={handlePress}
        onLayout={handleLayout}
        activeOpacity={0.8}
      >
        {children}
      </TouchableOpacity>

      <Modal
        visible={visible}
        transparent
        animationType="fade"
        onRequestClose={() => setVisible(false)}
      >
        <TouchableOpacity
          style={styles.overlay}
          activeOpacity={1}
          onPress={() => setVisible(false)}
        >
          <View
            style={[
              styles.tooltip,
              position === 'top' && styles.tooltipTop,
              position === 'bottom' && styles.tooltipBottom,
              position === 'left' && styles.tooltipLeft,
              position === 'right' && styles.tooltipRight,
            ]}
          >
            <Text style={styles.tooltipText}>{content}</Text>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tooltip: {
    backgroundColor: COLORS.text,
    padding: SPACING.sm,
    borderRadius: BORDER_RADIUS.md,
    maxWidth: 200,
  },
  tooltipTop: {
    marginBottom: SPACING.sm,
  },
  tooltipBottom: {
    marginTop: SPACING.sm,
  },
  tooltipLeft: {
    marginRight: SPACING.sm,
  },
  tooltipRight: {
    marginLeft: SPACING.sm,
  },
  tooltipText: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.surface,
  },
});

