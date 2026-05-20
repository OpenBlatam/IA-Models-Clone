import { S3_CLIENT_BUCKET_URL } from './aws-config';

export interface Course {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  instructor: string;
  duration: string;
  level: 'beginner' | 'intermediate' | 'advanced';
  price: number;
  classes: Class[];
}

export interface Class {
  id: string;
  title: string;
  duration: string;
  thumbnail: string;
  videoUrl: string;
  isLocked: boolean;
  isCompleted: boolean;
  progress: number;
  experience: number;
  description?: string;
  resources?: {
    title: string;
    url: string;
    type: 'pdf' | 'link' | 'video';
  }[];
}

// Curso de ChatGPT
export const chatGPTCourse: Course = {
  id: "chatgpt-course",
  title: "Master en ChatGPT y Generación de Contenido con IA",
  description: "Aprende a dominar ChatGPT y otras herramientas de IA para crear contenido profesional",
  thumbnail: `${S3_CLIENT_BUCKET_URL}/courses/chatgpt-thumbnail.jpg`,
  instructor: "Aníbal Rojas",
  duration: "10 horas",
  level: "intermediate",
  price: 99.99,
  classes: [
    {
      id: "1",
      title: "Explicación de cómo funciona",
      duration: "02:18",
      thumbnail: `${S3_CLIENT_BUCKET_URL}/thumbnails/1.jpg`,
      videoUrl: `${S3_CLIENT_BUCKET_URL}/videos/1.mp4`,
      isLocked: false,
      isCompleted: false,
      progress: 0,
      experience: 10
    },
    // ... resto de las clases del curso de ChatGPT
  ]
};

// Curso de Midjourney
export const midjourneyCourse: Course = {
  id: "midjourney-course",
  title: "Generación de Imágenes con Midjourney",
  description: "Domina la creación de imágenes con IA usando Midjourney",
  thumbnail: `${S3_CLIENT_BUCKET_URL}/courses/midjourney-thumbnail.jpg`,
  instructor: "Aníbal Rojas",
  duration: "8 horas",
  level: "intermediate",
  price: 79.99,
  classes: [
    {
      id: "mj-1",
      title: "Bienvenida a la generación de imágenes con Midjourney",
      duration: "01:40",
      thumbnail: `${S3_CLIENT_BUCKET_URL}/thumbnails/mj-1.jpg`,
      videoUrl: `${S3_CLIENT_BUCKET_URL}/videos/mj-1.mp4`,
      isLocked: false,
      isCompleted: false,
      progress: 0,
      experience: 10
    },
    // ... resto de las clases del curso de Midjourney
  ]
};

// Curso de RunwayML
export const runwayMLCourse: Course = {
  id: "runwayml-course",
  title: "Generación de Video con RunwayML",
  description: "Crea videos profesionales usando IA con RunwayML",
  thumbnail: `${S3_CLIENT_BUCKET_URL}/courses/runwayml-thumbnail.jpg`,
  instructor: "Aníbal Rojas",
  duration: "6 horas",
  level: "advanced",
  price: 89.99,
  classes: [
    {
      id: "rw-1",
      title: "Metas y Bienvenida",
      duration: "04:01",
      thumbnail: `${S3_CLIENT_BUCKET_URL}/thumbnails/rw-1.jpg`,
      videoUrl: `${S3_CLIENT_BUCKET_URL}/videos/rw-1.mp4`,
      isLocked: false,
      isCompleted: false,
      progress: 0,
      experience: 10
    },
    // ... resto de las clases del curso de RunwayML
  ]
};

// Exportar todos los cursos
export const courses: Course[] = [
  chatGPTCourse,
  midjourneyCourse,
  runwayMLCourse
];

// Función helper para obtener un curso por ID
export function getCourseById(courseId: string): Course | undefined {
  return courses.find(course => course.id === courseId);
}

// Función helper para obtener una clase específica
export function getClassById(courseId: string, classId: string): Class | undefined {
  const course = getCourseById(courseId);
  return course?.classes.find(cls => cls.id === classId);
} 