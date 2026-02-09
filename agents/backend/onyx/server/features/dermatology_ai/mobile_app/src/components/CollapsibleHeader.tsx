import React, { useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  Animated,
  TouchableOpacity,
  ViewStyle,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface CollapsibleHeaderProps {
  children: React.ReactNode;
  collapsedHeight?: number;
  expandedHeight?: number;
  scrollY: Animated.Value;
  threshold?: number;
  style?: ViewStyle;
}

const CollapsibleHeader: React.FC<CollapsibleHeaderProps> = ({
  children,
  collapsedHeight = 60,
  expandedHeight = 120,
  scrollY,
  threshold = 50,
  style,
}) => {
  const { colors } = useTheme();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const headerHeight = scrollY.interpolate({
    inputRange: [0, threshold],
    outputRange: [expandedHeight, collapsedHeight],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, threshold / 2, threshold],
    outputRange: [1, 0.7, 0.5],
    extrapolate: 'clamp',
  });

  React.useEffect(() => {
    const listener = scrollY.addListener(({ value }) => {
      setIsCollapsed(value > threshold);
    });

    return () => {
      scrollY.removeListener(listener);
    };
  }, [scrollY, threshold]);

  return (
    <Animated.View
      style={[
        styles.container,
        {
          backgroundColor: colors.card,
          height: headerHeight,
          opacity: headerOpacity,
        },
        isCollapsed && styles.collapsed,
        style,
      ]}
    >
      {children}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  collapsed: {
    paddingHorizontal: 16,
  },
});

export default CollapsibleHeader;

