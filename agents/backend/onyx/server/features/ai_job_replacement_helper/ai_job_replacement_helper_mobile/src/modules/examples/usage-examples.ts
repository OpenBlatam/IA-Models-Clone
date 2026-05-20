/**
 * Usage Examples for Modular Architecture
 * 
 * This file demonstrates how to use the modular structure
 * in your components and screens.
 */

// ============================================
// AUTH MODULE
// ============================================

import { authService, loginSchema, type LoginFormData } from '@/modules/auth';
import { handleError } from '@/modules/shared';

// Example: Login form
export async function exampleLogin(email: string, password: string) {
  try {
    const session = await authService.login({ email, password });
    // Handle success
    return session;
  } catch (error) {
    handleError(error, { showAlert: true });
    throw error;
  }
}

// Example: Using validation schema
import { useForm } from '@/hooks/useForm';

export function useLoginForm() {
  return useForm<LoginFormData>({
    initialValues: { email: '', password: '' },
    validationSchema: loginSchema,
    onSubmit: async (values) => {
      await authService.login(values);
    },
  });
}

// ============================================
// JOBS MODULE
// ============================================

import { jobService, useJobSearch, JobCard, type JobSwipeAction } from '@/modules/jobs';

// Example: Search jobs
export function useJobsExample() {
  const { data, isLoading } = useJobSearch({
    keywords: 'developer',
    location: 'Madrid',
    limit: 20,
  });

  return { jobs: data?.jobs || [], isLoading };
}

// Example: Swipe job
export async function exampleSwipeJob(userId: string, jobId: string) {
  try {
    await jobService.swipeJob(userId, {
      jobId,
      action: 'like',
    });
  } catch (error) {
    handleError(error);
  }
}

// Example: Using JobCard component
export function JobCardExample({ job }: { job: any }) {
  const handleSwipe = (action: JobSwipeAction) => {
    // Handle swipe
  };

  const handleApply = (jobId: string) => {
    // Handle apply
  };

  return <JobCard job={job} onSwipe={handleSwipe} onApply={handleApply} />;
}

// ============================================
// GAMIFICATION MODULE
// ============================================

import {
  gamificationService,
  calculateLevelProgress,
  type PointsAction,
} from '@/modules/gamification';

// Example: Get progress
export async function exampleGetProgress(userId: string) {
  try {
    const progress = await gamificationService.getProgress(userId);
    const levelProgress = calculateLevelProgress(progress.xp);
    return { progress, levelProgress };
  } catch (error) {
    handleError(error);
    throw error;
  }
}

// Example: Add points
export async function exampleAddPoints(userId: string, action: string) {
  try {
    const pointsAction: PointsAction = { action, amount: 10 };
    await gamificationService.addPoints(userId, pointsAction);
  } catch (error) {
    handleError(error);
  }
}

// ============================================
// DASHBOARD MODULE
// ============================================

import { dashboardService, QUICK_ACTIONS } from '@/modules/dashboard';

// Example: Get dashboard data
export async function exampleGetDashboard(userId: string) {
  try {
    const data = await dashboardService.getDashboard(userId);
    return data;
  } catch (error) {
    handleError(error);
    throw error;
  }
}

// Example: Using quick actions
export function QuickActionsExample() {
  return QUICK_ACTIONS.map((action) => (
    <Button key={action.id} onPress={() => navigate(action.route)}>
      {action.label}
    </Button>
  ));
}

// ============================================
// ROADMAP MODULE
// ============================================

import { roadmapService, STEP_STATUS } from '@/modules/roadmap';

// Example: Get roadmap
export async function exampleGetRoadmap(userId: string) {
  try {
    const roadmap = await roadmapService.getRoadmap(userId);
    return roadmap;
  } catch (error) {
    handleError(error);
    throw error;
  }
}

// Example: Complete step
export async function exampleCompleteStep(userId: string, stepId: string) {
  try {
    await roadmapService.completeStep(userId, stepId, 'Completed successfully');
  } catch (error) {
    handleError(error);
  }
}

// ============================================
// NOTIFICATIONS MODULE
// ============================================

import { notificationService, type NotificationFilters } from '@/modules/notifications';

// Example: Get notifications
export async function exampleGetNotifications(userId: string) {
  try {
    const filters: NotificationFilters = {
      unreadOnly: true,
      limit: 20,
    };
    const notifications = await notificationService.getNotifications(userId, filters);
    return notifications;
  } catch (error) {
    handleError(error);
    throw error;
  }
}

// ============================================
// SHARED MODULE
// ============================================

import {
  formatCurrency,
  formatDate,
  formatRelativeTime,
  validateEmail,
  sanitizeInput,
} from '@/modules/shared';

// Example: Formatting
export function formattingExamples() {
  const price = formatCurrency(1000); // "$1,000.00"
  const date = formatDate(new Date()); // "Jan 15, 2024"
  const relative = formatRelativeTime(new Date()); // "2 hours ago"
  return { price, date, relative };
}

// Example: Validation
export function validationExamples(email: string, input: string) {
  const isValidEmail = validateEmail(email);
  const sanitized = sanitizeInput(input);
  return { isValidEmail, sanitized };
}

// ============================================
// COMBINING MODULES
// ============================================

// Example: Complete flow using multiple modules
export async function exampleCompleteFlow(userId: string) {
  try {
    // Get dashboard data
    const dashboard = await dashboardService.getDashboard(userId);

    // Get jobs
    const jobs = await jobService.searchJobs(userId, { keywords: 'developer' });

    // Get progress
    const progress = await gamificationService.getProgress(userId);

    // Get notifications
    const notifications = await notificationService.getNotifications(userId, {
      unreadOnly: true,
    });

    return {
      dashboard,
      jobs: jobs.jobs,
      progress,
      notifications,
    };
  } catch (error) {
    handleError(error, { showAlert: true, logError: true });
    throw error;
  }
}


