import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface TooltipProps {
  children: React.ReactNode;
  text: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
}

const Tooltip: React.FC<TooltipProps> = ({
  children,
  text,
  position = 'top',
  delay = 0,
}) => {
  const { colors } = useTheme();
  const [visible, setVisible] = useState(false);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  const showTooltip = () => {
    if (delay > 0) {
      const id = setTimeout(() => setVisible(true), delay);
      setTimeoutId(id);
    } else {
      setVisible(true);
    }
  };

  const hideTooltip = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
    setVisible(false);
  };

  const getPositionStyles = () => {
    switch (position) {
      case 'top':
        return { bottom: '100%', marginBottom: 8 };
      case 'bottom':
        return { top: '100%', marginTop: 8 };
      case 'left':
        return { right: '100%', marginRight: 8 };
      case 'right':
        return { left: '100%', marginLeft: 8 };
      default:
        return { bottom: '100%', marginBottom: 8 };
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        onPressIn={showTooltip}
        onPressOut={hideTooltip}
        activeOpacity={1}
      >
        {children}
      </TouchableOpacity>
      {visible && (
        <View
          style={[
            styles.tooltip,
            {
              backgroundColor: colors.surface,
              ...getPositionStyles(),
            },
          ]}
        >
          <Text style={[styles.tooltipText, { color: colors.text }]}>
            {text}
          </Text>
          <View
            style={[
              styles.arrow,
              {
                borderTopColor: position === 'bottom' ? colors.surface : 'transparent',
                borderBottomColor: position === 'top' ? colors.surface : 'transparent',
                borderLeftColor: position === 'right' ? colors.surface : 'transparent',
                borderRightColor: position === 'left' ? colors.surface : 'transparent',
              },
            ]}
          />
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  tooltip: {
    position: 'absolute',
    padding: 8,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
    zIndex: 1000,
  },
  tooltipText: {
    fontSize: 12,
  },
  arrow: {
    position: 'absolute',
    width: 0,
    height: 0,
    borderWidth: 6,
  },
});

export default Tooltip;

