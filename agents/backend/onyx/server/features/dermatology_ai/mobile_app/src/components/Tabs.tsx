import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface Tab {
  label: string;
  value: string;
  icon?: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (value: string) => void;
  variant?: 'default' | 'pills' | 'underline';
}

const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onTabChange,
  variant = 'default',
}) => {
  const { colors } = useTheme();
  const [layout, setLayout] = React.useState<Record<string, { x: number; width: number }>>({});
  const indicatorPosition = useSharedValue(0);
  const indicatorWidth = useSharedValue(0);

  React.useEffect(() => {
    const activeLayout = layout[activeTab];
    if (activeLayout) {
      indicatorPosition.value = withSpring(activeLayout.x);
      indicatorWidth.value = withSpring(activeLayout.width);
    }
  }, [activeTab, layout, indicatorPosition, indicatorWidth]);

  const indicatorStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: indicatorPosition.value }],
      width: indicatorWidth.value,
    };
  });

  const getVariantStyles = () => {
    switch (variant) {
      case 'pills':
        return {
          container: { backgroundColor: colors.surface, borderRadius: 12, padding: 4 },
          tab: { borderRadius: 8 },
        };
      case 'underline':
        return {
          container: { borderBottomWidth: 1, borderBottomColor: colors.border },
          tab: {},
        };
      default:
        return {
          container: {},
          tab: {},
        };
    }
  };

  const variantStyles = getVariantStyles();

  return (
    <View
      style={[
        styles.container,
        variantStyles.container,
      ]}
    >
      {tabs.map((tab, index) => {
        const isActive = activeTab === tab.value;

        return (
          <TouchableOpacity
            key={tab.value}
            onPress={() => onTabChange(tab.value)}
            onLayout={(e) => {
              const { x, width } = e.nativeEvent.layout;
              setLayout((prev) => ({ ...prev, [tab.value]: { x, width } }));
            }}
            style={[
              styles.tab,
              variantStyles.tab,
              {
                backgroundColor:
                  variant === 'pills' && isActive ? colors.primary : 'transparent',
              },
            ]}
            activeOpacity={0.7}
          >
            {tab.icon && <View style={styles.icon}>{tab.icon}</View>}
            <Text
              style={[
                styles.tabText,
                {
                  color:
                    variant === 'pills'
                      ? isActive
                        ? '#fff'
                        : colors.text
                      : isActive
                      ? colors.primary
                      : colors.textSecondary,
                  fontWeight: isActive ? '600' : '400',
                },
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
      {variant === 'underline' && (
        <Animated.View
          style={[
            styles.indicator,
            {
              backgroundColor: colors.primary,
            },
            indicatorStyle,
          ]}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    position: 'relative',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  icon: {
    marginRight: 8,
  },
  tabText: {
    fontSize: 14,
    textAlign: 'center',
  },
  indicator: {
    position: 'absolute',
    bottom: 0,
    height: 2,
    borderRadius: 1,
  },
});

export default Tabs;
