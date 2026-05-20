"use client";

import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Clock, Lock, CheckCircle2, PlayCircle } from "lucide-react";
import { Academy, AcademyClass } from "@/lib/types/academy";

interface AcademyClassesModalProps {
  open: boolean;
  onClose: () => void;
  academy: Academy;
  onSelectClass: (classId: string) => void;
}

export default function AcademyClassesModal({
  open,
  onClose,
  academy,
  onSelectClass,
}: AcademyClassesModalProps) {
  const [classes, setClasses] = useState<AcademyClass[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchClasses = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const response = await fetch(`/api/academies/${academy.id}/classes`);
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Error al cargar las clases");
        }

        const data = await response.json();
        setClasses(data);
      } catch (error) {
        setError(error instanceof Error ? error.message : "Error al cargar las clases");
      } finally {
        setIsLoading(false);
      }
    };

    if (open) {
      fetchClasses();
    }
  }, [academy.id, open]);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">
            {academy.name}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Academy Info */}
          <div className="flex items-center gap-4 p-4 bg-muted rounded-lg">
            <img
              src={academy.thumbnail}
              alt={academy.name}
              className="w-24 h-24 object-cover rounded-lg"
            />
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="secondary">{academy.level}</Badge>
                <Badge variant="outline">{academy.category}</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                {academy.description}
              </p>
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  <span>{academy.totalDuration}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span>{academy.totalClasses} clases</span>
                </div>
              </div>
            </div>
          </div>

          {/* Classes List */}
          <ScrollArea className="h-[400px] pr-4">
            {isLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              </div>
            ) : error ? (
              <div className="text-center text-red-500 p-4">
                <p>{error}</p>
                <Button
                  variant="outline"
                  onClick={() => window.location.reload()}
                  className="mt-2"
                >
                  Reintentar
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                {classes.map((classItem) => (
                  <div
                    key={classItem.id}
                    className="flex items-center gap-4 p-4 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors"
                    onClick={() => onSelectClass(classItem.id)}
                  >
                    <div className="relative w-40 aspect-video">
                      <img
                        src={classItem.thumbnail}
                        alt={classItem.title}
                        className="w-full h-full object-cover rounded-lg"
                      />
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                        <PlayCircle className="w-8 h-8 text-white" />
                      </div>
                    </div>
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center justify-between">
                        <h3 className="font-medium">{classItem.title}</h3>
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm text-muted-foreground">
                            {classItem.duration}
                          </span>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {classItem.description}
                      </p>
                      <div className="flex items-center gap-2">
                        {classItem.isCompleted ? (
                          <Badge variant="secondary" className="bg-green-100 text-green-800 border-green-200">
                            <CheckCircle2 className="w-4 h-4 mr-1" />
                            Completado
                          </Badge>
                        ) : (
                          <Badge variant="secondary">
                            <Lock className="w-4 h-4 mr-1" />
                            {classItem.progress ?? 0}% Completado
                          </Badge>
                        )}
                        <Badge variant="outline">
                          {classItem.experience} XP
                        </Badge>
                      </div>
                      {(classItem.progress ?? 0) > 0 && !classItem.isCompleted && (
                        <Progress value={classItem.progress ?? 0} className="h-1" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>
      </DialogContent>
    </Dialog>
  );
}      