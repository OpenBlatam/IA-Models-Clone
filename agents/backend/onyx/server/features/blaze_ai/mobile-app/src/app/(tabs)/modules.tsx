import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';
import { ModuleCard } from '@/components/modules/module-card';

interface Module {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  type: string;
  version: string;
  icon: string;
  metrics: {
    cpu: number;
    memory: number;
    uptime: number;
  };
}

export default function ModulesScreen(): JSX.Element {
  const { theme } = useTheme();
  const { width, height } = useWindowDimensions();
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const styles = useMemo(() => createStyles(theme, width, height), [theme, width, height]);

  const modules: Module[] = [
    {
      id: 'cache',
      name: 'Cache Module',
      description: 'High-performance caching with multiple strategies',
      status: 'active',
      type: 'Performance',
      version: '2.1.0',
      icon: 'flash-outline',
      metrics: { cpu: 15, memory: 25, uptime: 99.8 },
    },
    {
      id: 'monitoring',
      name: 'Monitoring Module',
      description: 'Real-time system monitoring and alerting',
      status: 'active',
      type: 'Observability',
      version: '1.9.2',
      icon: 'analytics-outline',
      metrics: { cpu: 8, memory: 18, uptime: 99.9 },
    },
    {
      id: 'optimization',
      name: 'Optimization Module',
      description: 'AI-powered optimization algorithms',
      status: 'active',
      type: 'AI/ML',
      version: '3.0.1',
      icon: 'trending-up-outline',
      metrics: { cpu: 45, memory: 32, uptime: 98.5 },
    },
    {
      id: 'storage',
      name: 'Storage Module',
      description: 'Intelligent data storage and compression',
      status: 'active',
      type: 'Data',
      version: '2.4.0',
      icon: 'hardware-chip-outline',
      metrics: { cpu: 12, memory: 28, uptime: 99.7 },
    },
    {
      id: 'execution',
      name: 'Execution Module',
      description: 'Task execution and load balancing',
      status: 'active',
      type: 'Performance',
      version: '2.0.3',
      icon: 'play-circle-outline',
      metrics: { cpu: 22, memory: 35, uptime: 99.6 },
    },
    {
      id: 'engines',
      name: 'Engines Module',
      description: 'Quantum and neural optimization engines',
      status: 'active',
      type: 'AI/ML',
      version: '1.8.5',
      icon: 'rocket-outline',
      metrics: { cpu: 38, memory: 42, uptime: 97.8 },
    },
    {
      id: 'ml',
      name: 'ML Module',
      description: 'Machine learning model management',
      status: 'active',
      type: 'AI/ML',
      version: '2.2.1',
      icon: 'brain-outline',
      metrics: { cpu: 55, memory: 48, uptime: 96.2 },
    },
    {
      id: 'data-analysis',
      name: 'Data Analysis Module',
      description: 'Advanced data processing and analytics',
      status: 'active',
      type: 'Data',
      version: '1.7.8',
      icon: 'bar-chart-outline',
      metrics: { cpu: 28, memory: 31, uptime: 99.1 },
    },
    {
      id: 'ai-intelligence',
      name: 'AI Intelligence Module',
      description: 'Natural language and computer vision',
      status: 'active',
      type: 'AI/ML',
      version: '2.3.0',
      icon: 'eye-outline',
      metrics: { cpu: 42, memory: 38, uptime: 98.9 },
    },
    {
      id: 'api-rest',
      name: 'API REST Module',
      description: 'RESTful API interface and documentation',
      status: 'active',
      type: 'Integration',
      version: '2.1.5',
      icon: 'globe-outline',
      metrics: { cpu: 18, memory: 22, uptime: 99.5 },
    },
    {
      id: 'security',
      name: 'Security Module',
      description: 'Advanced security and encryption',
      status: 'active',
      type: 'Security',
      version: '1.9.8',
      icon: 'shield-checkmark-outline',
      metrics: { cpu: 5, memory: 12, uptime: 99.9 },
    },
    {
      id: 'distributed',
      name: 'Distributed Processing Module',
      description: 'Distributed computing and scaling',
      status: 'active',
      type: 'Performance',
      version: '2.0.7',
      icon: 'git-network-outline',
      metrics: { cpu: 35, memory: 29, uptime: 98.7 },
    },
  ];

  const categories = [
    { id: 'all', name: 'All', count: modules.length },
    { id: 'performance', name: 'Performance', count: modules.filter(m => m.type === 'Performance').length },
    { id: 'ai-ml', name: 'AI/ML', count: modules.filter(m => m.type === 'AI/ML').length },
    { id: 'data', name: 'Data', count: modules.filter(m => m.type === 'Data').length },
    { id: 'observability', name: 'Observability', count: modules.filter(m => m.type === 'Observability').length },
    { id: 'security', name: 'Security', count: modules.filter(m => m.type === 'Security').length },
    { id: 'integration', name: 'Integration', count: modules.filter(m => m.type === 'Integration').length },
  ];

  const filteredModules = selectedCategory === 'all' 
    ? modules 
    : modules.filter(m => m.type.toLowerCase().replace('/', '-') === selectedCategory);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>System Modules</Text>
          <Text style={styles.subtitle}>Manage and monitor Blaze AI modules</Text>
        </View>

        {/* Category Filter */}
        <View style={styles.categoriesContainer}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.categoriesScroll}
          >
            {categories.map((category) => (
              <TouchableOpacity
                key={category.id}
                style={[
                  styles.categoryButton,
                  selectedCategory === category.id && styles.categoryButtonActive,
                ]}
                onPress={() => setSelectedCategory(category.id)}
              >
                <Text
                  style={[
                    styles.categoryText,
                    selectedCategory === category.id && styles.categoryTextActive,
                  ]}
                >
                  {category.name}
                </Text>
                <View style={styles.categoryCount}>
                  <Text style={styles.categoryCountText}>{category.count}</Text>
                </View>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Modules Grid */}
        <View style={styles.modulesContainer}>
          {filteredModules.map((module) => (
            <ModuleCard
              key={module.id}
              module={module}
              onPress={() => console.log(`Navigate to ${module.name}`)}
            />
          ))}
        </View>

        {/* System Summary */}
        <View style={styles.summaryContainer}>
          <Text style={styles.summaryTitle}>System Summary</Text>
          <View style={styles.summaryGrid}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>{modules.length}</Text>
              <Text style={styles.summaryLabel}>Total Modules</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {modules.filter(m => m.status === 'active').length}
              </Text>
              <Text style={styles.summaryLabel}>Active</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {modules.filter(m => m.status === 'error').length}
              </Text>
              <Text style={styles.summaryLabel}>Errors</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {modules.filter(m => m.status === 'maintenance').length}
              </Text>
              <Text style={styles.summaryLabel}>Maintenance</Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(theme: any, width: number, height: number) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollView: {
      flex: 1,
    },
    scrollContent: {
      paddingBottom: theme.spacing.xxl,
    },
    header: {
      padding: theme.spacing.lg,
      paddingBottom: theme.spacing.md,
    },
    title: {
      fontSize: theme.typography.h1.fontSize,
      fontWeight: theme.typography.h1.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    subtitle: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.textSecondary,
    },
    categoriesContainer: {
      marginBottom: theme.spacing.lg,
    },
    categoriesScroll: {
      paddingHorizontal: theme.spacing.lg,
    },
    categoryButton: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.borderRadius.md,
      marginRight: theme.spacing.sm,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    categoryButtonActive: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary,
    },
    categoryText: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.text,
      fontWeight: '500',
    },
    categoryTextActive: {
      color: theme.colors.background,
    },
    categoryCount: {
      backgroundColor: theme.colors.background,
      paddingHorizontal: theme.spacing.xs,
      paddingVertical: 2,
      borderRadius: theme.borderRadius.sm,
      marginLeft: theme.spacing.sm,
    },
    categoryCountText: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.text,
      fontWeight: '600',
    },
    modulesContainer: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    summaryContainer: {
      paddingHorizontal: theme.spacing.lg,
    },
    summaryTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    summaryGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
    },
    summaryItem: {
      alignItems: 'center',
      flex: 1,
    },
    summaryValue: {
      fontSize: theme.typography.h3.fontSize,
      fontWeight: theme.typography.h3.fontWeight,
      color: theme.colors.primary,
      marginBottom: theme.spacing.xs,
    },
    summaryLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
  });
}
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';
import { ModuleCard } from '@/components/modules/module-card';

interface Module {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  type: string;
  version: string;
  icon: string;
  metrics: {
    cpu: number;
    memory: number;
    uptime: number;
  };
}

export default function ModulesScreen(): JSX.Element {
  const { theme } = useTheme();
  const { width, height } = useWindowDimensions();
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const styles = useMemo(() => createStyles(theme, width, height), [theme, width, height]);

  const modules: Module[] = [
    {
      id: 'cache',
      name: 'Cache Module',
      description: 'High-performance caching with multiple strategies',
      status: 'active',
      type: 'Performance',
      version: '2.1.0',
      icon: 'flash-outline',
      metrics: { cpu: 15, memory: 25, uptime: 99.8 },
    },
    {
      id: 'monitoring',
      name: 'Monitoring Module',
      description: 'Real-time system monitoring and alerting',
      status: 'active',
      type: 'Observability',
      version: '1.9.2',
      icon: 'analytics-outline',
      metrics: { cpu: 8, memory: 18, uptime: 99.9 },
    },
    {
      id: 'optimization',
      name: 'Optimization Module',
      description: 'AI-powered optimization algorithms',
      status: 'active',
      type: 'AI/ML',
      version: '3.0.1',
      icon: 'trending-up-outline',
      metrics: { cpu: 45, memory: 32, uptime: 98.5 },
    },
    {
      id: 'storage',
      name: 'Storage Module',
      description: 'Intelligent data storage and compression',
      status: 'active',
      type: 'Data',
      version: '2.4.0',
      icon: 'hardware-chip-outline',
      metrics: { cpu: 12, memory: 28, uptime: 99.7 },
    },
    {
      id: 'execution',
      name: 'Execution Module',
      description: 'Task execution and load balancing',
      status: 'active',
      type: 'Performance',
      version: '2.0.3',
      icon: 'play-circle-outline',
      metrics: { cpu: 22, memory: 35, uptime: 99.6 },
    },
    {
      id: 'engines',
      name: 'Engines Module',
      description: 'Quantum and neural optimization engines',
      status: 'active',
      type: 'AI/ML',
      version: '1.8.5',
      icon: 'rocket-outline',
      metrics: { cpu: 38, memory: 42, uptime: 97.8 },
    },
    {
      id: 'ml',
      name: 'ML Module',
      description: 'Machine learning model management',
      status: 'active',
      type: 'AI/ML',
      version: '2.2.1',
      icon: 'brain-outline',
      metrics: { cpu: 55, memory: 48, uptime: 96.2 },
    },
    {
      id: 'data-analysis',
      name: 'Data Analysis Module',
      description: 'Advanced data processing and analytics',
      status: 'active',
      type: 'Data',
      version: '1.7.8',
      icon: 'bar-chart-outline',
      metrics: { cpu: 28, memory: 31, uptime: 99.1 },
    },
    {
      id: 'ai-intelligence',
      name: 'AI Intelligence Module',
      description: 'Natural language and computer vision',
      status: 'active',
      type: 'AI/ML',
      version: '2.3.0',
      icon: 'eye-outline',
      metrics: { cpu: 42, memory: 38, uptime: 98.9 },
    },
    {
      id: 'api-rest',
      name: 'API REST Module',
      description: 'RESTful API interface and documentation',
      status: 'active',
      type: 'Integration',
      version: '2.1.5',
      icon: 'globe-outline',
      metrics: { cpu: 18, memory: 22, uptime: 99.5 },
    },
    {
      id: 'security',
      name: 'Security Module',
      description: 'Advanced security and encryption',
      status: 'active',
      type: 'Security',
      version: '1.9.8',
      icon: 'shield-checkmark-outline',
      metrics: { cpu: 5, memory: 12, uptime: 99.9 },
    },
    {
      id: 'distributed',
      name: 'Distributed Processing Module',
      description: 'Distributed computing and scaling',
      status: 'active',
      type: 'Performance',
      version: '2.0.7',
      icon: 'git-network-outline',
      metrics: { cpu: 35, memory: 29, uptime: 98.7 },
    },
  ];

  const categories = [
    { id: 'all', name: 'All', count: modules.length },
    { id: 'performance', name: 'Performance', count: modules.filter(m => m.type === 'Performance').length },
    { id: 'ai-ml', name: 'AI/ML', count: modules.filter(m => m.type === 'AI/ML').length },
    { id: 'data', name: 'Data', count: modules.filter(m => m.type === 'Data').length },
    { id: 'observability', name: 'Observability', count: modules.filter(m => m.type === 'Observability').length },
    { id: 'security', name: 'Security', count: modules.filter(m => m.type === 'Security').length },
    { id: 'integration', name: 'Integration', count: modules.filter(m => m.type === 'Integration').length },
  ];

  const filteredModules = selectedCategory === 'all' 
    ? modules 
    : modules.filter(m => m.type.toLowerCase().replace('/', '-') === selectedCategory);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>System Modules</Text>
          <Text style={styles.subtitle}>Manage and monitor Blaze AI modules</Text>
        </View>

        {/* Category Filter */}
        <View style={styles.categoriesContainer}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.categoriesScroll}
          >
            {categories.map((category) => (
              <TouchableOpacity
                key={category.id}
                style={[
                  styles.categoryButton,
                  selectedCategory === category.id && styles.categoryButtonActive,
                ]}
                onPress={() => setSelectedCategory(category.id)}
              >
                <Text
                  style={[
                    styles.categoryText,
                    selectedCategory === category.id && styles.categoryTextActive,
                  ]}
                >
                  {category.name}
                </Text>
                <View style={styles.categoryCount}>
                  <Text style={styles.categoryCountText}>{category.count}</Text>
                </View>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Modules Grid */}
        <View style={styles.modulesContainer}>
          {filteredModules.map((module) => (
            <ModuleCard
              key={module.id}
              module={module}
              onPress={() => console.log(`Navigate to ${module.name}`)}
            />
          ))}
        </View>

        {/* System Summary */}
        <View style={styles.summaryContainer}>
          <Text style={styles.summaryTitle}>System Summary</Text>
          <View style={styles.summaryGrid}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>{modules.length}</Text>
              <Text style={styles.summaryLabel}>Total Modules</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {modules.filter(m => m.status === 'active').length}
              </Text>
              <Text style={styles.summaryLabel}>Active</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {modules.filter(m => m.status === 'error').length}
              </Text>
              <Text style={styles.summaryLabel}>Errors</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {modules.filter(m => m.status === 'maintenance').length}
              </Text>
              <Text style={styles.summaryLabel}>Maintenance</Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(theme: any, width: number, height: number) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollView: {
      flex: 1,
    },
    scrollContent: {
      paddingBottom: theme.spacing.xxl,
    },
    header: {
      padding: theme.spacing.lg,
      paddingBottom: theme.spacing.md,
    },
    title: {
      fontSize: theme.typography.h1.fontSize,
      fontWeight: theme.typography.h1.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    subtitle: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.textSecondary,
    },
    categoriesContainer: {
      marginBottom: theme.spacing.lg,
    },
    categoriesScroll: {
      paddingHorizontal: theme.spacing.lg,
    },
    categoryButton: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.borderRadius.md,
      marginRight: theme.spacing.sm,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    categoryButtonActive: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary,
    },
    categoryText: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.text,
      fontWeight: '500',
    },
    categoryTextActive: {
      color: theme.colors.background,
    },
    categoryCount: {
      backgroundColor: theme.colors.background,
      paddingHorizontal: theme.spacing.xs,
      paddingVertical: 2,
      borderRadius: theme.borderRadius.sm,
      marginLeft: theme.spacing.sm,
    },
    categoryCountText: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.text,
      fontWeight: '600',
    },
    modulesContainer: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    summaryContainer: {
      paddingHorizontal: theme.spacing.lg,
    },
    summaryTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    summaryGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
    },
    summaryItem: {
      alignItems: 'center',
      flex: 1,
    },
    summaryValue: {
      fontSize: theme.typography.h3.fontSize,
      fontWeight: theme.typography.h3.fontWeight,
      color: theme.colors.primary,
      marginBottom: theme.spacing.xs,
    },
    summaryLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
  });
}


