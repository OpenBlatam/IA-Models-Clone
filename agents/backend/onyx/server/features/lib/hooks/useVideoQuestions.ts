import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { useSession } from 'next-auth/react';

interface User {
  id: string;
  name: string;
  image: string;
}

interface Question {
  id: string;
  content: string;
  createdAt: string;
  updatedAt: string;
  status: 'pending' | 'answered' | 'error';
  answer?: string;
  user: User;
}

interface UseVideoQuestionsReturn {
  questions: Question[];
  isLoading: boolean;
  error: string | null;
  askQuestion: (content: string) => Promise<void>;
}

export function useVideoQuestions(
  videoId: string,
  courseId: string
): UseVideoQuestionsReturn {
  const { data: session } = useSession();
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(
          `/api/questions?videoId=${videoId}&courseId=${courseId}`
        );
        if (!response.ok) {
          throw new Error('Error al cargar las preguntas');
        }
        const data = await response.json();
        setQuestions(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error desconocido');
      } finally {
        setIsLoading(false);
      }
    };

    fetchQuestions();
  }, [videoId, courseId]);

  const askQuestion = async (content: string) => {
    if (!session?.user) {
      throw new Error('Debes iniciar sesión para hacer preguntas');
    }

    try {
      const response = await fetch('/api/questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          videoId,
          courseId,
        }),
      });

      if (!response.ok) {
        throw new Error('Error al enviar la pregunta');
      }

      const newQuestion = await response.json();
      setQuestions((prev) => [newQuestion, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
      throw err;
    }
  };

  return {
    questions,
    isLoading,
    error,
    askQuestion,
  };
} 