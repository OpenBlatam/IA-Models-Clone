import type { ViewStyle, TextStyle, ImageStyle } from 'react-native';

export interface BaseComponentProps {
  testID?: string;
  accessibilityLabel?: string;
  accessibilityRole?: string;
  accessibilityHint?: string;
}

export interface StyledComponentProps {
  style?: ViewStyle | TextStyle | ImageStyle;
  className?: string;
}

export interface ButtonProps extends BaseComponentProps, StyledComponentProps {
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  fullWidth?: boolean;
  children: React.ReactNode;
}

export interface CardProps extends BaseComponentProps, StyledComponentProps {
  children: React.ReactNode;
  onPress?: () => void;
  delay?: number;
  elevated?: boolean;
}

export interface LoadingSpinnerProps extends BaseComponentProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
  message?: string;
}

export interface ErrorMessageProps extends BaseComponentProps {
  message: string;
  onRetry?: () => void;
  retryLabel?: string;
}

export interface EmptyStateProps extends BaseComponentProps {
  title: string;
  message?: string;
  icon?: string;
  actionLabel?: string;
  onAction?: () => void;
}

