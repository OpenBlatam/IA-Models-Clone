import React from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';
import { Course } from '@/lib/courses';
import { Button } from '@/components/ui/button';
import { Play, Clock, Star } from 'lucide-react';

interface CourseCarouselProps {
  courses: Course[];
  onSelectCourse: (courseId: string) => void;
  title?: string;
  subtitle?: string;
}

const CourseCarousel: React.FC<CourseCarouselProps> = ({
  courses,
  onSelectCourse,
  title = "Cursos Disponibles",
  subtitle = "Explora nuestra selección de cursos"
}) => {
  return (
    <div className="w-full py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold mb-2">{title}</h2>
          <p className="text-muted-foreground">{subtitle}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {courses.map((course) => (
            <motion.div
              key={course.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ y: -5 }}
              className="bg-card rounded-xl overflow-hidden shadow-lg border border-border"
            >
              <div className="relative aspect-video">
                <Image
                  src={course.thumbnail}
                  alt={course.title}
                  fill
                  className="object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <Button
                  variant="secondary"
                  size="sm"
                  className="absolute bottom-4 left-4 bg-white/10 backdrop-blur-sm hover:bg-white/20"
                  onClick={() => onSelectCourse(course.id)}
                >
                  <Play className="w-4 h-4 mr-2" />
                  Ver Curso
                </Button>
              </div>

              <div className="p-4">
                <h3 className="text-xl font-semibold mb-2">{course.title}</h3>
                <p className="text-muted-foreground text-sm mb-4">
                  {course.description}
                </p>

                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{course.duration}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Star className="w-4 h-4" />
                      <span>{course.level}</span>
                    </div>
                  </div>
                  <div className="font-semibold">
                    ${course.price}
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-border">
                  <div className="flex items-center gap-2">
                    <Image
                      src="/images/instructor-avatar.jpg"
                      alt={course.instructor}
                      width={24}
                      height={24}
                      className="rounded-full"
                    />
                    <span className="text-sm">{course.instructor}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CourseCarousel;    