import axios, { AxiosError } from 'axios';
import type {
  InspectionResult,
  CameraInfo,
  CameraSettings,
  DetectionSettings,
  Alert,
  AlertStatistics,
  BatchInspectionRequest,
  ReportRequest,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/quality-control`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const message =
      error.response?.data && typeof error.response.data === 'object' && 'message' in error.response.data
        ? (error.response.data as { message: string }).message
        : error.message || 'An error occurred';
    
    return Promise.reject(new Error(message));
  }
);

export const qualityControlApi = {
  // Camera operations
  async initializeCamera(): Promise<boolean> {
    const response = await apiClient.post('/camera/initialize');
    return response.data.success;
  },

  async startInspection(): Promise<boolean> {
    const response = await apiClient.post('/inspection/start');
    return response.data.success;
  },

  async stopInspection(): Promise<void> {
    await apiClient.post('/inspection/stop');
  },

  async getCameraInfo(): Promise<CameraInfo> {
    const response = await apiClient.get('/camera/info');
    return response.data;
  },

  async updateCameraSettings(settings: CameraSettings): Promise<void> {
    await apiClient.put('/camera/settings', settings);
  },

  // Inspection operations
  async inspectFrame(image?: string): Promise<InspectionResult> {
    const response = await apiClient.post('/inspection/inspect', {
      image,
    });
    return response.data;
  },

  async inspectBatch(request: BatchInspectionRequest): Promise<InspectionResult[]> {
    const response = await apiClient.post('/inspection/batch', request);
    return response.data;
  },

  async captureFrame(): Promise<string> {
    const response = await apiClient.get('/camera/capture', {
      responseType: 'blob',
    });
    const blob = response.data;
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
  },

  // Detection settings
  async updateDetectionSettings(settings: DetectionSettings): Promise<void> {
    await apiClient.put('/detection/settings', settings);
  },

  // Alerts
  async getRecentAlerts(level?: string, limit = 50): Promise<Alert[]> {
    const params = new URLSearchParams();
    if (level) params.append('level', level);
    params.append('limit', limit.toString());
    const response = await apiClient.get(`/alerts/recent?${params.toString()}`);
    return response.data;
  },

  async getAlertStatistics(): Promise<AlertStatistics> {
    const response = await apiClient.get('/alerts/statistics');
    return response.data;
  },

  // Reports
  async generateReport(request: ReportRequest): Promise<Blob> {
    const response = await apiClient.post('/reports/generate', request, {
      responseType: 'blob',
    });
    return response.data;
  },

  // Video analysis
  async analyzeVideo(videoFile: File, frameSkip = 5, maxFrames = 1000): Promise<InspectionResult[]> {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('frame_skip', frameSkip.toString());
    formData.append('max_frames', maxFrames.toString());
    const response = await apiClient.post('/video/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

