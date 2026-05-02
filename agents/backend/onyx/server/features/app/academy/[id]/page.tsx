"use client";

import { getAcademyById } from "@/lib/academies";
import { ClassCard } from "@/components/academy/class-card";
import { VideoPlayer } from "@/components/academy/video-player";
import { notFound } from "next/navigation";
import { useState } from "react";

interface AcademyPageProps {
  params: Promise<{
    id: string;
  }>;
}

export default async function AcademyPage({ params }: AcademyPageProps) {
  const resolvedParams = await params;
  const academy = getAcademyById(resolvedParams.id);

  if (!academy) {
    notFound();
  }

  const [selectedClassId, setSelectedClassId] = useState(
    academy.classes?.[0]?.id || ""
  );

  const selectedClass = academy.classes?.find(
    (c) => c.id === selectedClassId
  );

  const handleClassComplete = () => {
    // Aquí implementaremos la lógica para marcar la clase como completada
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
            <div className="relative aspect-video">
              <img
                src={academy.thumbnail}
                alt={academy.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="p-6">
              <h1 className="text-3xl font-bold mb-4">{academy.name}</h1>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                {academy.description}
              </p>
              <div className="flex flex-wrap gap-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center gap-2">
                  <span>Instructor:</span>
                  <span className="font-medium">{academy.instructor}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>Categoría:</span>
                  <span className="font-medium">{academy.category}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>Nivel:</span>
                  <span className="font-medium capitalize">{academy.level}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>Duración:</span>
                  <span className="font-medium">{academy.totalDuration}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>Clases:</span>
                  <span className="font-medium">{academy.totalClasses}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>Experiencia:</span>
                  <span className="font-medium">{academy.experience} XP</span>
                </div>
              </div>
            </div>
          </div>
          {selectedClass && (
            <div className="mt-8">
              <VideoPlayer
                classData={selectedClass}
                onComplete={handleClassComplete}
              />
            </div>
          )}
        </div>
        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4">Clases</h2>
          {academy.classes?.map((classData) => (
            <ClassCard
              key={classData.id}
              classData={classData}
              onSelect={setSelectedClassId}
              isSelected={classData.id === selectedClassId}
            />
          ))}
        </div>
      </div>
    </div>
  );
}      