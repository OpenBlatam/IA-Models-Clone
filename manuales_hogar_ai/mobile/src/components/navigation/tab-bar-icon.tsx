/**
 * Tab Bar Icon
 * ============
 * Icon component for tab bar navigation
 */

import { Ionicons } from '@expo/vector-icons';
import { ComponentProps } from 'react';

type IconName = ComponentProps<typeof Ionicons>['name'];

interface TabBarIconProps {
  name: IconName;
  color: string;
  size?: number;
}

export function TabBarIcon({ name, color, size = 24 }: TabBarIconProps) {
  return <Ionicons name={name} size={size} color={color} />;
}




