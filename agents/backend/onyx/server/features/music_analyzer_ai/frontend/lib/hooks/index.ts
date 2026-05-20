/**
 * Shared hooks exports.
 * Centralized export point for all shared custom hooks.
 */

export { useDebounce } from './use-debounce';
export { useLocalStorage } from './use-local-storage';
export type {
  UseLocalStorageOptions,
  UseLocalStorageReturn,
} from './use-local-storage';
export {
  useMediaQuery,
  useBreakpoints,
  useIsMobile,
  useIsTablet,
  useIsDesktop,
  useIsDarkMode,
  usePrefersReducedMotion,
} from './use-media-query';
export { useApiHealth } from './use-api-health';
export type { ApiHealthStatus } from './use-api-health';
export { useFormValidation } from './use-form-validation';
export type {
  UseFormValidationOptions,
  UseFormValidationReturn,
  ValidationResult,
} from './use-form-validation';
export {
  useQueryWithErrorHandling,
  useMutationWithHandling,
  useOptimisticMutation,
} from './use-react-query';
export { useStoreSubscription } from './use-store-subscription';
export {
  useMemoizedCallback,
  useMemoizedCallbacks,
  useStableValue,
} from './use-memoized-callbacks';
export { useSafeAction } from './use-safe-action';
export type { UseSafeActionOptions, UseSafeActionReturn } from './use-safe-action';
export { useKeyboardShortcuts } from './use-keyboard-shortcuts';
export type {
  KeyboardShortcut,
  UseKeyboardShortcutsOptions,
} from './use-keyboard-shortcuts';
export { MUSIC_SHORTCUTS } from './use-keyboard-shortcuts';
export { useViewport } from './use-viewport';
export type {
  ViewportSize,
  BreakpointStatus,
  UseViewportOptions,
} from './use-viewport';
export { BREAKPOINTS } from './use-viewport';
export { useOnlineStatus } from './use-online-status';
export type { OnlineStatus } from './use-online-status';
export { useIntersectionObserver } from './use-intersection-observer';
export type {
  UseIntersectionObserverOptions,
  UseIntersectionObserverReturn,
} from './use-intersection-observer';
export { useClickOutside } from './use-click-outside';
export type { UseClickOutsideOptions } from './use-click-outside';
export { usePrevious } from './use-previous';
export { useAsync } from './use-async';
export type {
  UseAsyncOptions,
  UseAsyncReturn,
} from './use-async';
export { useQueryParams } from './use-query-params';
export type {
  UseQueryParamsOptions,
  UseQueryParamsReturn,
} from './use-query-params';
export { useWindowSize } from './use-window-size';
export type {
  WindowSize,
  UseWindowSizeOptions,
} from './use-window-size';
export { useToggle } from './use-toggle';
export type { UseToggleReturn } from './use-toggle';
export { useScroll } from './use-scroll';
export type {
  ScrollPosition,
  UseScrollOptions,
} from './use-scroll';
export { useScrollToTop } from './use-scroll-to-top';
export type {
  UseScrollToTopOptions,
  UseScrollToTopReturn,
} from './use-scroll-to-top';
export { useCopyToClipboard } from './use-copy-to-clipboard';
export type { UseCopyToClipboardReturn } from './use-copy-to-clipboard';
export { useFocusTrap } from './use-focus-trap';
export type { UseFocusTrapOptions } from './use-focus-trap';
export { useModal } from './use-modal';
export type { UseModalReturn } from './use-modal';
export { useTooltip } from './use-tooltip';
export type {
  UseTooltipOptions,
  UseTooltipReturn,
} from './use-tooltip';
export { useId } from './use-id';
export { useErrorHandler } from './use-error-handler';
export type { UseErrorHandlerReturn } from './use-error-handler';
export { useHover } from './use-hover';
export type { UseHoverReturn } from './use-hover';
export { useLongPress } from './use-long-press';
export type { UseLongPressOptions } from './use-long-press';
export { useCountdown } from './use-countdown';
export type {
  UseCountdownOptions,
  UseCountdownReturn,
} from './use-countdown';
export { useInterval } from './use-interval';
export type { UseIntervalOptions } from './use-interval';
export { useTimeout } from './use-timeout';
export type { UseTimeoutOptions } from './use-timeout';
export { useMount } from './use-mount';
export type { UseMountOptions } from './use-mount';
export { useUpdateEffect } from './use-update-effect';
export { useCookie } from './use-cookie';
export type {
  UseCookieOptions,
  UseCookieReturn,
} from './use-cookie';
export { useFetch } from './use-fetch';
export type {
  UseFetchOptions,
  UseFetchReturn,
} from './use-fetch';
export { useEventListener } from './use-event-listener';
export type { UseEventListenerOptions } from './use-event-listener';
export { useNetworkStatus } from './use-network-status';
export type { NetworkStatus } from './use-network-status';
export { useBattery } from './use-battery';
export type { BatteryStatus } from './use-battery';
export { useDevice } from './use-device';
export type { DeviceInfo } from './use-device';
export { useStorageAdvanced } from './use-storage-advanced';
export type {
  UseStorageAdvancedOptions,
  UseStorageAdvancedReturn,
} from './use-storage-advanced';
export { useForm } from './use-form';
export type {
  UseFormOptions,
  UseFormReturn,
  FormState,
} from './use-form';
export { useSearch } from './use-search';
export type {
  UseSearchOptions,
  UseSearchReturn,
} from './use-search';
export { usePagination } from './use-pagination';
export type {
  UsePaginationOptions,
  UsePaginationReturn,
} from './use-pagination';
export { useSort } from './use-sort';
export type {
  UseSortOptions,
  UseSortReturn,
  SortConfig,
} from './use-sort';
export { useThrottleValue } from './use-throttle-value';
export type { UseThrottleValueOptions } from './use-throttle-value';
export { useDebounceValue } from './use-debounce-value';
export type { UseDebounceValueOptions } from './use-debounce-value';
export { useMemoCompare } from './use-memo-compare';
export { useIsomorphicLayoutEffect } from './use-isomorphic-layout-effect';
export { useCache } from './use-cache';
export type { UseCacheOptions } from './use-cache';
export { useDeepCompareEffect } from './use-deep-compare-effect';
export { useDeepCompareMemo } from './use-deep-compare-memo';
export { useRaf } from './use-raf';
export { useStateWithHistory } from './use-state-with-history';
export type { UseStateWithHistoryOptions } from './use-state-with-history';
export { useStateWithValidator } from './use-state-with-validator';
export type {
  Validator,
  UseStateWithValidatorOptions,
} from './use-state-with-validator';
export { useEventEmitter } from './use-event-emitter';
export { usePromise } from './use-promise';
export type { PromiseState } from './use-promise';
export { useObservable, useObservableState } from './use-observable';
export { useWorker } from './use-worker';
export type { UseWorkerOptions } from './use-worker';
export { useHash, useHashString } from './use-hash';
export { useMemoize } from './use-memoize';
export type { UseMemoizeOptions } from './use-memoize';
export { useBatch, useBatchAsync } from './use-batch';
export type { UseBatchOptions } from './use-batch';
export { useRateLimit } from './use-rate-limit';
export type { UseRateLimitOptions } from './use-rate-limit';
export { useSemaphore } from './use-semaphore';
export { useStream } from './use-stream';
export { useReactive, useComputed } from './use-reactive';
export { useStateMachine } from './use-state-machine';
export { usePipeline } from './use-pipeline';
export { useCircuitBreaker } from './use-circuit-breaker';
export { useRetryAdvanced, useCreateRetryFunction } from './use-retry-advanced';
export { usePerformanceMonitor } from './use-performance-monitor';
export { useAnalytics, usePageView } from './use-analytics';
export {
  useMemoizedValue,
  useStableCallback,
  useDebouncedCallback,
  useThrottledCallback,
  useRenderCount,
} from './use-performance-optimization';
export type {
  UseMemoizedValueOptions,
  UseDebouncedCallbackOptions,
  UseThrottledCallbackOptions,
} from './use-performance-optimization';

export {
  useAnnounce,
  useFocusTrap,
  usePrefersReducedMotion,
  usePrefersHighContrast,
  useKeyboardNavigation,
} from './use-accessibility';
export type {
  UseAnnounceOptions,
  UseFocusTrapOptions,
} from './use-accessibility';
