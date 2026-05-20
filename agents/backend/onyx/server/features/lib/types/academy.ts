export interface Academy {
  id: string;
  name: string;
  description: string;
  thumbnail: string;
  instructor: string;
  instructorAvatar?: string;
  instructorSubscribers?: number;
  category: string;
  level: 'beginner' | 'intermediate' | 'advanced';
  totalClasses: number;
  totalDuration: string;
  experience: number;
  totalProgress?: number;
  s3Config: {
    bucketName: string;
    clientKey: string;
    region: string;
  };
  classes: AcademyClass[];
  createdAt: string;
  updatedAt: string;
  rating?: number;
  difficulty: "beginner" | "intermediate" | "advanced";
  tags: string[];
}

export interface AcademyClass {
  id: string;
  academyId: string;
  title: string;
  description: string;
  videoUrl: string;
  thumbnail: string;
  duration: string;
  order: number;
  isLocked?: boolean;
  isCompleted?: boolean;
  progress?: number;
  experience: number;
  views?: number;
  resources?: Resource[];
  createdAt: string;
  updatedAt: string;
}

export interface Resource {
  id: string;
  type: 'pdf' | 'link' | 'code';
  title: string;
  url: string;
}

export interface AcademyProgress {
  id: string;
  userId: string;
  academyId: string;
  currentClassId: string;
  completedClasses: string[];
  totalProgress: number;
  lastAccessed: string;
  createdAt: string;
  updatedAt: string;
}  