"use client";

import { cn } from "@/lib/utils";

interface InstructorInfoProps {
  name: string;
  className?: string;
}

export function InstructorInfo({ name, className }: InstructorInfoProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <img
        src={`https://ui-avatars.com/api/?name=${encodeURIComponent(
          name
        )}&background=random`}
        alt={name}
        className="w-10 h-10 rounded-full flex-shrink-0"
      />
      <div className="min-w-0">
        <p className="text-sm font-medium truncate">{name}</p>
      </div>
    </div>
  );
} 