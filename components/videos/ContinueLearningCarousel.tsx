"use client";

import { useState } from "react";
import { X, PlayCircle } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import VideoPlayer from "./VideoPlayer";

interface CourseProgress {
  id: string;
  title: string;
  instructor?: string;
  image: string;
  currentClass: number;
  totalClasses: number;
  progress: number; // 0-100
  route: string;
  icon: React.ReactNode;
  videoUrl?: string;
}

interface ContinueLearningCarouselProps {
  userName?: string;
  onCourseClick?: (course: CourseProgress) => void;
}

const mockCourses: CourseProgress[] = [
  {
    id: "soft-skills",
    title: "Desarrollo de Habilidades Blandas: Métodos y Práctica",
    instructor: undefined,
    image: "/images/soft-skills.jpg",
    currentClass: 4,
    totalClasses: 15,
    progress: 28,
    route: "/dashboard/videos?path=soft-skills",
    videoUrl: "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    icon: <span className="bg-pink-600 text-white rounded-full p-1"><svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3.5 6.5l-2.5 2.5-2.5-2.5C6.5 13.5 5 11.5 5 9a7 7 0 0 1 7-7Z"/></svg></span>
  },
  {
    id: "ai-meetings",
    title: "Uso de IA en Reuniones: Técnicas y Herramientas",
    instructor: "Aníbal Rojas",
    image: "/images/ai-meetings.jpg",
    currentClass: 4,
    totalClasses: 15,
    progress: 30,
    route: "/dashboard/videos?path=ai-meetings",
    videoUrl: "https://storage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    icon: <span className="bg-green-700 text-white rounded-full p-1">AI</span>
  },
  {
    id: "english-basic",
    title: "Curso de Inglés Básico para Profesionales",
    instructor: undefined,
    image: "/images/english-basic.jpg",
    currentClass: 1,
    totalClasses: 12,
    progress: 8,
    route: "/dashboard/videos?path=english-basic",
    videoUrl: "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    icon: <span className="bg-purple-700 text-white rounded-full p-1"><svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3.5 6.5l-2.5 2.5-2.5-2.5C6.5 13.5 5 11.5 5 9a7 7 0 0 1 7-7Z"/></svg></span>
  }
];

export default function ContinueLearningCarousel({ userName = "Raul", onCourseClick }: ContinueLearningCarouselProps) {
  const [courses, setCourses] = useState<CourseProgress[]>(mockCourses);
  const [selectedCourse, setSelectedCourse] = useState<CourseProgress | null>(null);
  const router = useRouter();

  const handleRemove = (id: string) => {
    setCourses((prev) => prev.filter((c) => c.id !== id));
  };

  const handleCardClick = (course: CourseProgress) => {
    onCourseClick?.(course);
  };

  const handleCloseVideo = () => {
    setSelectedCourse(null);
  };

  return (
    <section className="w-full mb-10">
      <div className="flex items-center justify-between mb-4 px-2">
        <h2 className="text-2xl md:text-3xl font-extrabold text-foreground tracking-tight drop-shadow">{userName}, continúa aprendiendo</h2>
        <Link href="/dashboard/courses" className="text-primary underline font-semibold text-base md:text-lg">Ir a mis cursos &rarr;</Link>
      </div>
      <div className="overflow-x-auto pb-2">
        <div className="flex gap-8 min-w-[600px]">
          {courses.map((course) => (
            <div
              key={course.id}
              className="relative bg-zinc-900/80 backdrop-blur-lg border border-zinc-700 rounded-2xl shadow-2xl min-w-[320px] max-w-xs flex-shrink-0 cursor-pointer transition-transform duration-200 hover:scale-105 hover:shadow-2xl group"
              style={{ boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.25)" }}
              onClick={() => handleCardClick(course)}
            >
              {/* Close button */}
              <button
                className="absolute top-3 right-3 z-10 bg-zinc-800/80 hover:bg-zinc-700 rounded-full p-1"
                onClick={e => { e.stopPropagation(); handleRemove(course.id); }}
                aria-label="Quitar curso"
              >
                <X className="w-6 h-6 text-zinc-300" />
              </button>
              {/* Image and play button */}
              <div className="relative h-48 rounded-t-2xl overflow-hidden flex items-center justify-center">
                <Image src={course.image} alt={course.title} fill className="object-cover group-hover:scale-110 transition-transform duration-300" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="bg-white/90 rounded-full p-3 shadow-xl border-4 border-primary group-hover:scale-110 transition-transform">
                    <PlayCircle className="w-14 h-14 text-primary" />
                  </span>
                </div>
                {course.instructor && (
                  <span className="absolute bottom-2 left-2 bg-zinc-800/80 text-white text-xs px-2 py-1 rounded-md shadow">
                    {course.instructor}
                  </span>
                )}
              </div>
              {/* Progress and info */}
              <div className="flex items-center gap-4 px-5 py-4">
                {/* Circular progress */}
                <div className="relative w-12 h-12 flex items-center justify-center">
                  <svg className="absolute top-0 left-0" width="48" height="48">
                    <circle cx="24" cy="24" r="20" stroke="#27272a" strokeWidth="5" fill="none" />
                    <circle
                      cx="24"
                      cy="24"
                      r="20"
                      stroke="#22d3ee"
                      strokeWidth="5"
                      fill="none"
                      strokeDasharray={2 * Math.PI * 20}
                      strokeDashoffset={2 * Math.PI * 20 * (1 - course.progress / 100)}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                    {course.icon}
                  </span>
                </div>
                <div className="flex flex-col">
                  <div className="text-xs text-zinc-400 font-medium mb-1">Clase {course.currentClass} de {course.totalClasses}</div>
                  <div className="text-lg font-bold text-white truncate max-w-[170px] leading-tight drop-shadow-sm">{course.title}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Video Player Modal */}
      {selectedCourse && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="relative w-full max-w-5xl">
            <button
              onClick={handleCloseVideo}
              className="absolute -top-12 right-0 bg-zinc-800 hover:bg-zinc-700 rounded-full p-2 text-white"
            >
              <X className="w-6 h-6" />
            </button>
            <VideoPlayer courseId={selectedCourse.id} autoPlay />
          </div>
        </div>
      )}
    </section>
  );
}   