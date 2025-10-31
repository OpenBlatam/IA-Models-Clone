import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Types for the store
interface Post {
  id: string;
  topic: string;
  content: string;
  targetAudience: string;
  industry: string;
  tone: 'professional' | 'casual' | 'friendly';
  postType: 'announcement' | 'educational' | 'update' | 'insight';
  status: 'draft' | 'published' | 'scheduled';
  createdAt: Date;
  updatedAt: Date;
  optimizationScore: number;
  suggestions: string[];
  keywords: string[];
}

interface PostGenerationRequest {
  topic: string;
  keyPoints: string[];
  targetAudience: string;
  industry: string;
  tone: 'professional' | 'casual' | 'friendly';
  postType: 'announcement' | 'educational' | 'update' | 'insight';
  keywords?: string[];
  additionalContext?: string;
}

interface PostState {
  // State
  posts: Post[];
  currentPost: Post | null;
  isLoading: boolean;
  error: string | null;
  filters: {
    status: string;
    tone: string;
    postType: string;
  };
  
  // Actions
  generatePost: (request: PostGenerationRequest) => Promise<Post>;
  updatePost: (id: string, updates: Partial<Post>) => void;
  deletePost: (id: string) => void;
  setCurrentPost: (post: Post | null) => void;
  setFilters: (filters: Partial<PostState['filters']>) => void;
  clearError: () => void;
  resetState: () => void;
}

// Initial state
const initialState = {
  posts: [],
  currentPost: null,
  isLoading: false,
  error: null,
  filters: {
    status: '',
    tone: '',
    postType: '',
  },
};

// Zustand store with middleware
export const usePostStore = create<PostState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Generate new post
        generatePost: async (request: PostGenerationRequest) => {
          set({ isLoading: true, error: null });
          
          try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            const newPost: Post = {
              id: `post_${Date.now()}`,
              topic: request.topic,
              content: `🚀 ${request.topic}\n\n${request.keyPoints.map(point => `• ${point}`).join('\n')}\n\n#${request.industry} #Innovation #Growth`,
              targetAudience: request.targetAudience,
              industry: request.industry,
              tone: request.tone,
              postType: request.postType,
              status: 'draft',
              createdAt: new Date(),
              updatedAt: new Date(),
              optimizationScore: Math.random() * 0.3 + 0.7, // 0.7-1.0
              suggestions: ['Add more hashtags', 'Include a call-to-action'],
              keywords: request.keywords || [],
            };

            set(state => ({
              posts: [newPost, ...state.posts],
              currentPost: newPost,
              isLoading: false,
            }));

            return newPost;
          } catch (error) {
            set({ 
              error: error instanceof Error ? error.message : 'Failed to generate post',
              isLoading: false 
            });
            throw error;
          }
        },

        // Update existing post
        updatePost: (id: string, updates: Partial<Post>) => {
          set(state => ({
            posts: state.posts.map(post =>
              post.id === id
                ? { ...post, ...updates, updatedAt: new Date() }
                : post
            ),
            currentPost: state.currentPost?.id === id
              ? { ...state.currentPost, ...updates, updatedAt: new Date() }
              : state.currentPost,
          }));
        },

        // Delete post
        deletePost: (id: string) => {
          set(state => ({
            posts: state.posts.filter(post => post.id !== id),
            currentPost: state.currentPost?.id === id ? null : state.currentPost,
          }));
        },

        // Set current post
        setCurrentPost: (post: Post | null) => {
          set({ currentPost: post });
        },

        // Update filters
        setFilters: (filters: Partial<PostState['filters']>) => {
          set(state => ({
            filters: { ...state.filters, ...filters },
          }));
        },

        // Clear error
        clearError: () => {
          set({ error: null });
        },

        // Reset state
        resetState: () => {
          set(initialState);
        },
      }),
      {
        name: 'linkedin-posts-storage',
        partialize: (state) => ({
          posts: state.posts,
          filters: state.filters,
        }),
      }
    ),
    {
      name: 'linkedin-posts-store',
    }
  )
);

// Selectors for optimized re-renders
export const usePosts = () => usePostStore(state => state.posts);
export const useCurrentPost = () => usePostStore(state => state.currentPost);
export const useIsLoading = () => usePostStore(state => state.isLoading);
export const useError = () => usePostStore(state => state.error);
export const useFilters = () => usePostStore(state => state.filters);

// Computed selectors
export const useFilteredPosts = () => {
  const posts = usePosts();
  const filters = useFilters();
  
  return posts.filter(post => {
    if (filters.status && post.status !== filters.status) return false;
    if (filters.tone && post.tone !== filters.tone) return false;
    if (filters.postType && post.postType !== filters.postType) return false;
    return true;
  });
};

export const usePostStats = () => {
  const posts = usePosts();
  
  return {
    total: posts.length,
    drafts: posts.filter(p => p.status === 'draft').length,
    published: posts.filter(p => p.status === 'published').length,
    scheduled: posts.filter(p => p.status === 'scheduled').length,
    avgOptimizationScore: posts.length > 0 
      ? posts.reduce((sum, p) => sum + p.optimizationScore, 0) / posts.length 
      : 0,
  };
};

// Export types
export type { Post, PostGenerationRequest, PostState }; 