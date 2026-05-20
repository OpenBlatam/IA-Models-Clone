import { useState } from "react";
import { ChevronRight, BookOpen, Video, Trophy, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

interface LearningPath {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  route: string;
  isActive?: boolean;
}

interface VideoLearningPathsProps {
  currentPath?: string | null;
}

const learningPaths: LearningPath[] = [
  {
    id: "current",
    title: "Current Course",
    description: "Continue with your current learning path",
    icon: <Video className="w-5 h-5" />,
    route: "/dashboard/videos",
    isActive: true,
  },
  {
    id: "courses",
    title: "All Courses",
    description: "Browse all available courses",
    icon: <BookOpen className="w-5 h-5" />,
    route: "/dashboard/courses",
  },
  {
    id: "achievements",
    title: "Achievements",
    description: "Track your learning progress",
    icon: <Trophy className="w-5 h-5" />,
    route: "/dashboard/achievements",
  },
  {
    id: "community",
    title: "Community",
    description: "Connect with other learners",
    icon: <Users className="w-5 h-5" />,
    route: "/dashboard/community",
  },
];

export default function VideoLearningPaths({ currentPath }: VideoLearningPathsProps) {
  const [selectedPath, setSelectedPath] = useState<string>(currentPath || "current");

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-foreground">Learning Paths</h2>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {learningPaths.map((path) => (
          <Link
            key={path.id}
            href={path.route}
            onClick={() => setSelectedPath(path.id)}
            className={cn(
              "p-4 rounded-lg transition-all duration-200",
              "flex items-center gap-4",
              "border border-border/50",
              "hover:border-primary/50 hover:bg-primary/5",
              "focus:outline-none focus:ring-2 focus:ring-primary/50",
              selectedPath === path.id
                ? "bg-primary/10 border-primary/50"
                : "bg-background/50"
            )}
          >
            <div className="p-2 rounded-md bg-primary/10 text-primary">
              {path.icon}
            </div>
            <div className="flex-1 text-left">
              <h3 className="font-medium text-foreground">{path.title}</h3>
              <p className="text-sm text-muted-foreground">{path.description}</p>
            </div>
            <ChevronRight className="w-5 h-5 text-muted-foreground" />
          </Link>
        ))}
      </div>
    </div>
  );
} 