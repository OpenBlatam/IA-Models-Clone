import React, { useMemo } from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import type { RootStackParamList } from '../types/navigation';
import { NAVIGATION_CONFIG } from '../config/navigation';
import { TabNavigator } from './TabNavigator';

// Feature modules
import { RealTimeScanScreen } from '../features/real-time-scan';
import {
  AnalysisScreen,
  ReportScreen,
  ComparisonScreen,
} from '../features/analysis';
import { RecommendationsScreen } from '../features/recommendations';

const Stack = createStackNavigator<RootStackParamList>();

/**
 * Stack Navigator Component with all main screens
 */
export const StackNavigator: React.FC = React.memo(() => {
  const stackScreenOptions = useMemo(
    () => ({
      headerStyle: {
        backgroundColor: NAVIGATION_CONFIG.HEADER.BACKGROUND_COLOR,
      },
      headerTintColor: NAVIGATION_CONFIG.HEADER.TINT_COLOR,
      headerTitleStyle: NAVIGATION_CONFIG.HEADER.TITLE_STYLE,
    }),
    []
  );

  return (
    <Stack.Navigator screenOptions={stackScreenOptions}>
      <Stack.Screen
        name={NAVIGATION_CONFIG.SCREENS.MAIN_TABS.name}
        component={TabNavigator}
        options={NAVIGATION_CONFIG.SCREENS.MAIN_TABS.options}
      />
      <Stack.Screen
        name={NAVIGATION_CONFIG.SCREENS.REAL_TIME_SCAN.name}
        component={RealTimeScanScreen}
        options={{ title: NAVIGATION_CONFIG.SCREENS.REAL_TIME_SCAN.title }}
      />
      <Stack.Screen
        name={NAVIGATION_CONFIG.SCREENS.ANALYSIS.name}
        component={AnalysisScreen}
        options={{ title: NAVIGATION_CONFIG.SCREENS.ANALYSIS.title }}
      />
      <Stack.Screen
        name={NAVIGATION_CONFIG.SCREENS.RECOMMENDATIONS.name}
        component={RecommendationsScreen}
        options={{
          title: NAVIGATION_CONFIG.SCREENS.RECOMMENDATIONS.title,
        }}
      />
      <Stack.Screen
        name={NAVIGATION_CONFIG.SCREENS.REPORT.name}
        component={ReportScreen}
        options={{ title: NAVIGATION_CONFIG.SCREENS.REPORT.title }}
      />
      <Stack.Screen
        name={NAVIGATION_CONFIG.SCREENS.COMPARISON.name}
        component={ComparisonScreen}
        options={{ title: NAVIGATION_CONFIG.SCREENS.COMPARISON.title }}
      />
    </Stack.Navigator>
  );
});

StackNavigator.displayName = 'StackNavigator';

