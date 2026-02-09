/**
 * Modules Index
 * Central export for all feature modules
 */

// Feature Modules
export * as VideoModule from './video';
export * as AuthModule from './auth';
export * as TemplateModule from './template';
export * as AnalyticsModule from './analytics';
export * as SharedModule from './shared';

// Re-export commonly used items
export { VideoCard, VideoList, VideoProgress, VideoStatusBadge } from './video';
export { AuthGuard } from './auth';
export { TemplateCard } from './template';
export { QuotaIndicator } from './analytics';
export { EmptyState, ErrorState, SectionHeader } from './shared';

