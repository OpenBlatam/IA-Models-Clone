import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { PostCreate, MemeCreate, PlatformConnect, TemplateCreate } from '@/types';

// Posts hooks
export function usePosts(status?: string) {
  return useQuery({
    queryKey: ['posts', status],
    queryFn: () => api.posts.getAll(status),
  });
}

export function usePost(postId: string) {
  return useQuery({
    queryKey: ['post', postId],
    queryFn: () => api.posts.getById(postId),
    enabled: !!postId,
  });
}

export function useCreatePost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (post: PostCreate) => api.posts.create(post),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });
}

export function usePublishPost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (postId: string) => api.posts.publish(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });
}

export function useCancelPost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (postId: string) => api.posts.cancel(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });
}

// Memes hooks
export function useMemes(params?: { category?: string; tags?: string; query?: string }) {
  return useQuery({
    queryKey: ['memes', params],
    queryFn: () => api.memes.getAll(params),
  });
}

export function useRandomMeme(category?: string) {
  return useQuery({
    queryKey: ['randomMeme', category],
    queryFn: () => api.memes.getRandom(category),
  });
}

export function useCreateMeme() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (formData: FormData) => api.memes.create(formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memes'] });
    },
  });
}

export function useDeleteMeme() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (memeId: string) => api.memes.delete(memeId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memes'] });
    },
  });
}

export function useMemeCategories() {
  return useQuery({
    queryKey: ['memeCategories'],
    queryFn: () => api.memes.getCategories(),
  });
}

// Calendar hooks
export function useCalendarEvents(startDate?: string, endDate?: string) {
  return useQuery({
    queryKey: ['calendar', startDate, endDate],
    queryFn: () => api.calendar.getEvents(startDate, endDate),
  });
}

export function useDailyEvents(date: string) {
  return useQuery({
    queryKey: ['dailyEvents', date],
    queryFn: () => api.calendar.getDaily(date),
    enabled: !!date,
  });
}

export function useWeeklyEvents(startDate?: string) {
  return useQuery({
    queryKey: ['weeklyEvents', startDate],
    queryFn: () => api.calendar.getWeekly(startDate),
  });
}

// Platforms hooks
export function usePlatforms() {
  return useQuery({
    queryKey: ['platforms'],
    queryFn: () => api.platforms.getAll(),
  });
}

export function useConnectPlatform() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (connection: PlatformConnect) => api.platforms.connect(connection),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['platforms'] });
    },
  });
}

export function useDisconnectPlatform() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (platform: string) => api.platforms.disconnect(platform),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['platforms'] });
    },
  });
}

// Analytics hooks
export function usePlatformAnalytics(platform: string, days: number = 7) {
  return useQuery({
    queryKey: ['platformAnalytics', platform, days],
    queryFn: () => api.analytics.getPlatformAnalytics(platform, days),
    enabled: !!platform,
  });
}

export function usePostAnalytics(postId: string, platform?: string) {
  return useQuery({
    queryKey: ['postAnalytics', postId, platform],
    queryFn: () => api.analytics.getPostAnalytics(postId, platform),
    enabled: !!postId,
  });
}

export function useBestPerformingPosts(platform?: string, limit: number = 10) {
  return useQuery({
    queryKey: ['bestPerforming', platform, limit],
    queryFn: () => api.analytics.getBestPerforming(platform, limit),
  });
}

// Dashboard hooks
export function useDashboardOverview(days: number = 7) {
  return useQuery({
    queryKey: ['dashboardOverview', days],
    queryFn: () => api.dashboard.getOverview(days),
  });
}

export function useDashboardEngagement(days: number = 7) {
  return useQuery({
    queryKey: ['dashboardEngagement', days],
    queryFn: () => api.dashboard.getEngagement(days),
  });
}

export function useUpcomingPosts(limit: number = 10) {
  return useQuery({
    queryKey: ['upcomingPosts', limit],
    queryFn: () => api.dashboard.getUpcomingPosts(limit),
  });
}

// Templates hooks
export function useTemplates(params?: { query?: string; platform?: string; category?: string }) {
  return useQuery({
    queryKey: ['templates', params],
    queryFn: () => api.templates.getAll(params),
  });
}

export function useTemplate(templateId: string) {
  return useQuery({
    queryKey: ['template', templateId],
    queryFn: () => api.templates.getById(templateId),
    enabled: !!templateId,
  });
}

export function useCreateTemplate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (template: TemplateCreate) => api.templates.create(template),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['templates'] });
    },
  });
}

export function useDeleteTemplate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (templateId: string) => api.templates.delete(templateId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['templates'] });
    },
  });
}

export function useRenderTemplate() {
  return useMutation({
    mutationFn: ({ templateId, variables }: { templateId: string; variables: Record<string, string> }) =>
      api.templates.render(templateId, variables),
  });
}


