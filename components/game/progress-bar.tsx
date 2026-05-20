import { cn } from "@/lib/utils";

interface ProgressBarProps {
  currentExperience: number;
  nextLevelExperience: number;
  className?: string;
}

export function ProgressBar({
  currentExperience,
  nextLevelExperience,
  className,
}: ProgressBarProps) {
  const progress = (currentExperience / nextLevelExperience) * 100;

  return (
    <div className={cn("w-full", className)}>
      <div className="flex justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">
          Level {Math.floor(currentExperience / 1000) + 1}
        </span>
        <span className="text-sm font-medium text-gray-700">
          {currentExperience} / {nextLevelExperience} XP
        </span>
      </div>
      <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full transition-all duration-300 ease-in-out"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
} 