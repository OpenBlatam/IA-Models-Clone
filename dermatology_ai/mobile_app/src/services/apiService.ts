import axios, { AxiosError } from 'axios';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';
import * as FileSystem from 'expo-file-system';
import { AnalysisResult, Recommendations, HistoryItem, ApiResponse } from '../types';

interface AnalyzeImageOptions {
  enhance?: boolean;
  bodyArea?: string;
  userId?: string;
}

interface AnalyzeVideoOptions {
  userId?: string;
}

interface AnalyzeBodyAreaOptions {
  userId?: string;
}

interface GetRecommendationsOptions {
  includeRoutine?: boolean;
  userId?: string;
}

interface AnalysisData {
  imageUri?: string;
  analysisId?: string;
}

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    // const token = AsyncStorage.getItem('auth_token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

class ApiService {
  /**
   * Analyze image
   */
  async analyzeImage(
    imageUri: string,
    options: AnalyzeImageOptions = {}
  ): Promise<ApiResponse<AnalysisResult>> {
    try {
      const formData = new FormData();
      
      // Get file info
      const fileInfo = await FileSystem.getInfoAsync(imageUri);
      const filename = imageUri.split('/').pop() || 'image.jpg';
      const fileType = filename.split('.').pop() || 'jpg';
      
      formData.append('file', {
        uri: imageUri,
        type: `image/${fileType}`,
        name: filename,
      } as any);
      
      if (options.enhance !== undefined) {
        formData.append('enhance', options.enhance.toString());
      }
      if (options.bodyArea) {
        formData.append('body_area', options.bodyArea);
      }
      if (options.userId) {
        formData.append('user_id', options.userId);
      }
      
      const response = await apiClient.post<ApiResponse<AnalysisResult>>(
        API_ENDPOINTS.ANALYZE_IMAGE,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      return response.data;
    } catch (error) {
      console.error('Error analyzing image:', error);
      throw error;
    }
  }

  /**
   * Analyze video
   */
  async analyzeVideo(
    videoUri: string,
    options: AnalyzeVideoOptions = {}
  ): Promise<ApiResponse<AnalysisResult>> {
    try {
      const formData = new FormData();
      
      const filename = videoUri.split('/').pop() || 'video.mp4';
      const fileType = filename.split('.').pop() || 'mp4';
      
      formData.append('file', {
        uri: videoUri,
        type: `video/${fileType}`,
        name: filename,
      } as any);
      
      if (options.userId) {
        formData.append('user_id', options.userId);
      }
      
      const response = await apiClient.post<ApiResponse<AnalysisResult>>(
        API_ENDPOINTS.ANALYZE_VIDEO,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      return response.data;
    } catch (error) {
      console.error('Error analyzing video:', error);
      throw error;
    }
  }

  /**
   * Get recommendations
   */
  async getRecommendations(
    analysisData: AnalysisData,
    options: GetRecommendationsOptions = {}
  ): Promise<ApiResponse<Recommendations>> {
    try {
      const formData = new FormData();
      
      if (analysisData.imageUri) {
        const filename = analysisData.imageUri.split('/').pop() || 'image.jpg';
        const fileType = filename.split('.').pop() || 'jpg';
        
        formData.append('file', {
          uri: analysisData.imageUri,
          type: `image/${fileType}`,
          name: filename,
        } as any);
      }
      
      if (analysisData.analysisId) {
        formData.append('analysis_id', analysisData.analysisId);
      }
      
      if (options.includeRoutine !== undefined) {
        formData.append('include_routine', options.includeRoutine.toString());
      }
      if (options.userId) {
        formData.append('user_id', options.userId);
      }
      
      const response = await apiClient.post<ApiResponse<Recommendations>>(
        API_ENDPOINTS.GET_RECOMMENDATIONS,
        formData
      );
      
      return response.data;
    } catch (error) {
      console.error('Error getting recommendations:', error);
      throw error;
    }
  }

  /**
   * Get intelligent recommendations
   */
  async getIntelligentRecommendations(
    userId: string,
    skinData: Record<string, any>
  ): Promise<ApiResponse<Recommendations>> {
    try {
      const response = await apiClient.post<ApiResponse<Recommendations>>(
        API_ENDPOINTS.INTELLIGENT_RECOMMENDATIONS,
        {
          user_id: userId,
          skin_data: skinData,
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error getting intelligent recommendations:', error);
      throw error;
    }
  }

  /**
   * Get history
   */
  async getHistory(userId: string): Promise<ApiResponse<HistoryItem[]>> {
    try {
      const response = await apiClient.get<ApiResponse<HistoryItem[]>>(
        `${API_ENDPOINTS.HISTORY}/${userId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error getting history:', error);
      throw error;
    }
  }

  /**
   * Compare history records
   */
  async compareHistory(
    recordId1: string,
    recordId2: string
  ): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        `${API_ENDPOINTS.COMPARE_HISTORY}/${recordId1}/${recordId2}`
      );
      return response.data;
    } catch (error) {
      console.error('Error comparing history:', error);
      throw error;
    }
  }

  /**
   * Get timeline
   */
  async getTimeline(userId: string): Promise<ApiResponse<any[]>> {
    try {
      const response = await apiClient.get<ApiResponse<any[]>>(
        `${API_ENDPOINTS.TIMELINE}/${userId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error getting timeline:', error);
      throw error;
    }
  }

  /**
   * Get analytics
   */
  async getAnalytics(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        `${API_ENDPOINTS.ANALYTICS_USER}/${userId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error getting analytics:', error);
      throw error;
    }
  }

  /**
   * Get statistics
   */
  async getStatistics(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        `${API_ENDPOINTS.STATISTICS}/${userId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error getting statistics:', error);
      throw error;
    }
  }

  /**
   * Get report (JSON, PDF, or HTML)
   */
  async getReport(
    analysisId: string,
    format: 'json' | 'pdf' | 'html' = 'json'
  ): Promise<ApiResponse<any>> {
    try {
      const endpoint =
        format === 'json'
          ? API_ENDPOINTS.REPORT_JSON
          : format === 'pdf'
          ? API_ENDPOINTS.REPORT_PDF
          : API_ENDPOINTS.REPORT_HTML;
      
      const response = await apiClient.post<ApiResponse<any>>(endpoint, {
        analysis_id: analysisId,
      });
      
      return response.data;
    } catch (error) {
      console.error('Error getting report:', error);
      throw error;
    }
  }

  /**
   * Search products
   */
  async searchProducts(
    query: string,
    filters: Record<string, any> = {}
  ): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        API_ENDPOINTS.PRODUCTS_SEARCH,
        {
          params: {
            query,
            ...filters,
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error searching products:', error);
      throw error;
    }
  }

  /**
   * Compare products
   */
  async compareProducts(productIds: string[]): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.post<ApiResponse<any>>(
        API_ENDPOINTS.PRODUCTS_COMPARE,
        {
          product_ids: productIds,
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error comparing products:', error);
      throw error;
    }
  }

  /**
   * Analyze body area
   */
  async analyzeBodyArea(
    imageUri: string,
    bodyArea: string,
    options: AnalyzeBodyAreaOptions = {}
  ): Promise<ApiResponse<AnalysisResult>> {
    try {
      const formData = new FormData();
      
      const filename = imageUri.split('/').pop() || 'image.jpg';
      const fileType = filename.split('.').pop() || 'jpg';
      
      formData.append('file', {
        uri: imageUri,
        type: `image/${fileType}`,
        name: filename,
      } as any);
      
      formData.append('body_area', bodyArea);
      
      if (options.userId) {
        formData.append('user_id', options.userId);
      }
      
      const response = await apiClient.post<ApiResponse<AnalysisResult>>(
        API_ENDPOINTS.ANALYZE_BODY_AREA,
        formData
      );
      
      return response.data;
    } catch (error) {
      console.error('Error analyzing body area:', error);
      throw error;
    }
  }

  /**
   * Get alerts
   */
  async getAlerts(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        `${API_ENDPOINTS.ALERTS}/${userId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error getting alerts:', error);
      throw error;
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<ApiResponse<any>> {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        API_ENDPOINTS.HEALTH
      );
      return response.data;
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }
}

export default new ApiService();

