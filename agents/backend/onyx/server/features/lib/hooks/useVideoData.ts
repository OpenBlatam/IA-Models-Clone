import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { getAcademyById } from '@/lib/academies';

interface VideoData {
  video: {
    id: string;
    title: string;
    description: string;
    url: string;
    thumbnail: string;
    duration: string;
    progress: number;
    experience: number;
    isCompleted: boolean;
  };
  resources: {
    id: string;
    title: string;
    type: "file" | "reading";
    url: string;
  }[];
  comments: {
    id: string;
    content: string;
    createdAt: string;
    likes: number;
    isLiked: boolean;
    user: {
      name: string;
      image: string;
      email: string;
    };
    replies?: any[];
  }[];
}

export function useVideoData(videoId: string, courseId: string) {
  const [data, setData] = useState<VideoData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchVideoData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Get academy data
        const academy = getAcademyById(courseId);
        if (!academy || !academy.classes) {
          throw new Error('Academy not found');
        }

        // Find the class
        const classData = academy.classes.find(c => c.id === videoId);
        if (!classData) {
          throw new Error('Class not found');
        }

        // Use the video URL directly from the class object
        const videoUrl = classData.videoUrl;

        // Transform class data to match VideoData interface
        const videoData: VideoData = {
          video: {
            id: classData.id,
            title: classData.title,
            description: classData.description,
            url: videoUrl,
            thumbnail: classData.thumbnail,
            duration: classData.duration,
            progress: 0, // You might want to get this from user progress
            experience: classData.experience,
            isCompleted: false, // You might want to get this from user progress
          },
          resources: classData.resources ? JSON.parse(JSON.stringify(classData.resources)) : [],
          comments: [] // You might want to implement comments separately
        };

        setData(videoData);
      } catch (err) {
        console.error('Error loading video data:', err);
        setError('Error al cargar los datos del video');
        toast.error('Error al cargar los datos del video');
      } finally {
        setIsLoading(false);
      }
    };

    if (videoId && courseId) {
      fetchVideoData();
    }
  }, [videoId, courseId]);

  const updateProgress = async (progress: number) => {
    try {
      // You might want to implement progress tracking
      setData(prev => prev ? {
        ...prev,
        video: {
          ...prev.video,
          progress
        }
      } : null);
    } catch (err) {
      console.error('Error updating progress:', err);
      toast.error('Error al actualizar el progreso');
    }
  };

  const addComment = async (content: string) => {
    try {
      // You might want to implement comments
      const newComment = {
        id: Date.now().toString(),
        content,
        createdAt: new Date().toISOString(),
        likes: 0,
        isLiked: false,
        user: {
          name: 'User',
          image: '',
          email: ''
        }
      };

      setData(prev => prev ? {
        ...prev,
        comments: [newComment, ...prev.comments]
      } : null);
    } catch (err) {
      console.error('Error adding comment:', err);
      toast.error('Error al agregar el comentario');
    }
  };

  const addResource = async (resource: { title: string; type: "file" | "reading"; url: string }) => {
    try {
      // You might want to implement resources
      const newResource = {
        id: Date.now().toString(),
        ...resource
      };

      setData(prev => prev ? {
        ...prev,
        resources: [...prev.resources, newResource]
      } : null);
    } catch (err) {
      console.error('Error adding resource:', err);
      toast.error('Error al agregar el recurso');
    }
  };

  return {
    data,
    isLoading,
    error,
    updateProgress,
    addComment,
    addResource,
  };
}      