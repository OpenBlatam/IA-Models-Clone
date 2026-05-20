import { create } from 'zustand';
import { MAX_HISTORY_ITEMS, MAX_ALERTS_ITEMS } from '@/config/constants';
import type { InspectionResult } from '@/modules/inspection/types';
import type { Alert } from '@/modules/alerts/types';
import type { CameraInfo } from '@/modules/camera/types';

interface QualityControlState {
  isInspecting: boolean;
  currentResult: InspectionResult | null;
  inspectionHistory: InspectionResult[];
  alerts: Alert[];
  cameraInfo: CameraInfo | null;
  error: string | null;
  setInspecting: (isInspecting: boolean) => void;
  setCurrentResult: (result: InspectionResult | null) => void;
  addToHistory: (result: InspectionResult) => void;
  addAlert: (alert: Alert) => void;
  setAlerts: (alerts: Alert[]) => void;
  setCameraInfo: (info: CameraInfo | null) => void;
  setError: (error: string | null) => void;
  clearHistory: () => void;
}

export const useQualityControlStore = create<QualityControlState>((set) => ({
  isInspecting: false,
  currentResult: null,
  inspectionHistory: [],
  alerts: [],
  cameraInfo: null,
  error: null,
  setInspecting: (isInspecting) => set({ isInspecting }),
  setCurrentResult: (result) => set({ currentResult: result }),
  addToHistory: (result) =>
    set((state) => ({
      inspectionHistory: [result, ...state.inspectionHistory].slice(0, MAX_HISTORY_ITEMS),
    })),
  addAlert: (alert) =>
    set((state) => ({
      alerts: [alert, ...state.alerts].slice(0, MAX_ALERTS_ITEMS),
    })),
  setAlerts: (alerts) => set({ alerts }),
  setCameraInfo: (info) => set({ cameraInfo: info }),
  setError: (error) => set({ error }),
  clearHistory: () => set({ inspectionHistory: [] }),
}));

