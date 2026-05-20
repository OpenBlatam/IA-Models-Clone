export interface Exercise {
  id: string;
  type: string;
  question: string;
  options: string[];
  correctAnswer: string;
  points: number;
  lessonId: string;
  audioUrl?: string | null;
}

export interface Lesson {
  id: string;
  title: string;
  description: string;
  difficulty: string;
  category: string;
  requiredLevel: number;
  experienceReward: number;
  exercises: Exercise[];
  content?: {
    sections: {
      title: string;
      content: string;
    }[];
  } | null;
}

export interface LessonContentProps {
  lesson: Lesson;
  onComplete: () => void;
}

export interface MarketingSection {
  title: string;
  description: string;
  icon: 'globe' | 'share' | 'barChart' | 'target' | 'brain' | 'rocket';
  example: string;
}

export interface CompletedLesson {
  lessonId: string;
  completedAt: Date;
}

export interface Theme {
  background: string;
  card: string;
  border: string;
  text: {
    primary: string;
    secondary: string;
    accent: string;
  };
  button: {
    primary: string;
    secondary: string;
  };
  accent: {
    primary: string;
    secondary: string;
  };
  shadow: string;
  hoverShadow: string;
}

export interface Themes {
  light: Theme;
  dark: Theme;
} 