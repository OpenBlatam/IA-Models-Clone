import { AcademyClass } from "@/lib/types/academy";
import { ClassResources } from "./class-resources";
import { Play, CheckCircle } from "lucide-react";

interface ClassCardProps {
  classData: AcademyClass;
  onSelect: (classId: string) => void;
  isSelected: boolean;
}

export function ClassCard({ classData, onSelect, isSelected }: ClassCardProps) {
  return (
    <div
      className={`p-6 rounded-lg border ${
        isSelected
          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
          : "border-gray-200 dark:border-gray-700"
      }`}
    >
      <div className="flex items-start gap-4">
        <div className="relative w-32 h-20 rounded-md overflow-hidden">
          <img
            src={classData.thumbnail}
            alt={classData.title}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
            {classData.isCompleted ? (
              <CheckCircle className="h-8 w-8 text-white" />
            ) : (
              <Play className="h-8 w-8 text-white" />
            )}
          </div>
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold mb-2">{classData.title}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
            {classData.description}
          </p>
          <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>{classData.duration}</span>
            <span>•</span>
            <span>{classData.experience} XP</span>
          </div>
        </div>
      </div>
      {classData.resources && classData.resources.length > 0 && (
        <ClassResources resources={classData.resources} />
      )}
      <button
        onClick={() => onSelect(classData.id)}
        className={`mt-4 w-full py-2 px-4 rounded-md text-sm font-medium transition-colors ${
          isSelected
            ? "bg-blue-600 text-white hover:bg-blue-700"
            : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 hover:bg-gray-200 dark:hover:bg-gray-700"
        }`}
      >
        {isSelected ? "Clase Actual" : "Ver Clase"}
      </button>
    </div>
  );
} 