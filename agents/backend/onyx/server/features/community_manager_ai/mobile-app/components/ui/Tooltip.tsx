import { ReactNode, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ViewStyle } from 'react-native';
import Animated, { FadeIn, FadeOut } from 'react-native-reanimated';

interface TooltipProps {
  children: ReactNode;
  text: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  style?: ViewStyle;
}

export function Tooltip({ children, text, position = 'top', style }: TooltipProps) {
  const [visible, setVisible] = useState(false);

  const getPositionStyle = (): ViewStyle => {
    switch (position) {
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
    <View style={[styles.container, style]}>
      <TouchableOpacity
        onPressIn={() => setVisible(true)}
        onPressOut={() => setVisible(false)}
        activeOpacity={1}
      >
        {children}
      </TouchableOpacity>
      {visible && (
        <Animated.View
          entering={FadeIn}
          exiting={FadeOut}
          style={[styles.tooltip, getPositionStyle()]}
        >
          <Text style={styles.text}>{text}</Text>
        </Animated.View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  tooltip: {
    position: 'absolute',
    backgroundColor: '#1f2937',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    zIndex: 1000,
  },
  text: {
    color: '#fff',
    fontSize: 12,
  },
});

