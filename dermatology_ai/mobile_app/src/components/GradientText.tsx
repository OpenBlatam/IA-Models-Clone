import React from 'react';
import { Text, StyleSheet, TextStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import MaskedView from '@react-native-masked-view/masked-view';

interface GradientTextProps {
  children: React.ReactNode;
  colors: string[];
  style?: TextStyle;
  start?: { x: number; y: number };
  end?: { x: number; y: number };
}

const GradientText: React.FC<GradientTextProps> = ({
  children,
  colors,
  style,
  start = { x: 0, y: 0 },
  end = { x: 1, y: 0 },
}) => {
  return (
    <MaskedView
      style={styles.mask}
      maskElement={<Text style={[styles.text, style]}>{children}</Text>}
    >
      <LinearGradient colors={colors} start={start} end={end}>
        <Text style={[styles.text, style, { opacity: 0 }]}>{children}</Text>
      </LinearGradient>
    </MaskedView>
  );
};

const styles = StyleSheet.create({
  mask: {
    flexDirection: 'row',
  },
  text: {
    backgroundColor: 'transparent',
  },
});

export default GradientText;

