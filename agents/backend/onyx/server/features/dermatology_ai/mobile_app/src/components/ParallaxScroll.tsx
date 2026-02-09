import React, { useRef } from 'react';
import {
  ScrollView,
  ScrollViewProps,
  Animated,
  View,
  StyleSheet,
} from 'react-native';

interface ParallaxScrollProps extends ScrollViewProps {
  parallaxHeaderHeight: number;
  renderHeader: (scrollY: Animated.Value) => React.ReactNode;
  children: React.ReactNode;
}

const ParallaxScroll: React.FC<ParallaxScrollProps> = ({
  parallaxHeaderHeight,
  renderHeader,
  children,
  ...props
}) => {
  const scrollY = useRef(new Animated.Value(0)).current;

  const headerTranslateY = scrollY.interpolate({
    inputRange: [0, parallaxHeaderHeight],
    outputRange: [0, -parallaxHeaderHeight],
    extrapolate: 'clamp',
  });

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, parallaxHeaderHeight / 2, parallaxHeaderHeight],
    outputRange: [1, 0.5, 0],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.header,
          {
            height: parallaxHeaderHeight,
            transform: [{ translateY: headerTranslateY }],
            opacity: headerOpacity,
          },
        ]}
      >
        {renderHeader(scrollY)}
      </Animated.View>
      <ScrollView
        {...props}
        scrollEventThrottle={16}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
        contentContainerStyle={[
          { paddingTop: parallaxHeaderHeight },
          props.contentContainerStyle,
        ]}
      >
        {children}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1,
    overflow: 'hidden',
  },
});

export default ParallaxScroll;

