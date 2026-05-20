"use client";
import { motion, AnimatePresence } from "framer-motion";
import { X, PlayCircle, Lock, CheckCircle2, Award, Clock } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import * as Tooltip from "@radix-ui/react-tooltip";
import React, { useState, useRef } from "react";
import ReactPlayer from "react-player";
import { useRouter } from "next/navigation";

interface Class {
  id: string;
  title: string;
  duration: string;
  thumbnail: string;
  isLocked: boolean;
  isCompleted?: boolean;
  progress?: number;
  experience: number;
  videoUrl?: string;
  academyId: string;
}

interface ClassesModalProps {
  open: boolean;
  onClose: () => void;
  classes: Class[];
  currentClassId: string;
  onSelectClass: (classId: string) => void;
}

export default function ClassesModal({ open, onClose, classes, currentClassId, onSelectClass }: ClassesModalProps) {
  const [hoveredClassId, setHoveredClassId] = useState<string | null>(null);
  const videoRefs = useRef<{ [key: string]: HTMLVideoElement | null }>({});
  const router = useRouter();
  const totalProgress = classes.reduce((acc, curr) => acc + (curr.progress ?? 0), 0) / classes.length;
  const totalExperience = classes.reduce((acc, curr) => acc + curr.experience, 0);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-background rounded-2xl border border-border w-full max-w-2xl max-h-[80vh] overflow-hidden shadow-2xl"
          >
            <div className="p-4 border-b border-border flex items-center justify-between bg-muted/50 backdrop-blur-sm">
              <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                <Award className="w-5 h-5 text-primary" />
                Lista de Clases
              </h2>
              <Tooltip.Provider>
                <Tooltip.Root>
                  <Tooltip.Trigger asChild>
                    <button 
                      onClick={onClose} 
                      className="text-muted-foreground hover:text-foreground transition-colors duration-200 hover:scale-110"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </Tooltip.Trigger>
                  <Tooltip.Portal>
                    <Tooltip.Content
                      className="bg-background text-foreground px-3 py-2 rounded-lg text-sm shadow-lg border border-border backdrop-blur-sm"
                      sideOffset={5}
                    >
                      Cerrar
                      <Tooltip.Arrow className="fill-background" />
                    </Tooltip.Content>
                  </Tooltip.Portal>
                </Tooltip.Root>
              </Tooltip.Provider>
            </div>

            {/* Course Progress */}
            <div className="p-4 border-b border-border bg-muted/50 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-primary" />
                  <span className="text-sm font-medium text-foreground">Progreso del Curso</span>
                </div>
                <span className="text-sm text-muted-foreground">{Math.round(totalProgress)}%</span>
              </div>
              <Progress value={totalProgress} className="h-2 bg-muted" />
              <div className="mt-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Award className="w-4 h-4 text-primary" />
                  Total XP: {totalExperience}
                </span>
                <span className="text-xs text-muted-foreground">
                  {classes.filter(c => c.isCompleted).length} de {classes.length} clases completadas
                </span>
              </div>
            </div>

            <div className="overflow-y-auto max-h-[calc(80vh-200px)]">
              {classes.map((classItem, index) => (
                <motion.div
                  key={classItem.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`p-4 border-b border-border hover:bg-muted/50 transition-all duration-300 cursor-pointer ${
                    classItem.id === currentClassId ? "bg-muted/50" : ""
                  }`}
                  onClick={() => {
                    if (!classItem.isLocked) {
                      onSelectClass(classItem.id);
                      router.push(`/dashboard/videos?courseId=${classItem.academyId}&classId=${classItem.id}`);
                    }
                  }}
                >
                  <div className="flex items-center gap-4">
                    {/* Thumbnail/Preview */}
                    <motion.div 
                      className="relative w-24 h-16 rounded-lg overflow-hidden flex-shrink-0"
                      whileHover={{ scale: 1.05 }}
                      transition={{ duration: 0.2 }}
                      onMouseEnter={() => setHoveredClassId(classItem.id)}
                      onMouseLeave={() => setHoveredClassId(null)}
                    >
                      {hoveredClassId === classItem.id && !classItem.isLocked && classItem.videoUrl ? (
                        <ReactPlayer
                          url={classItem.videoUrl}
                          playing={true}
                          muted={true}
                          loop={true}
                          width="100%"
                          height="100%"
                          playsinline={true}
                          style={{ objectFit: 'cover', pointerEvents: 'none' }}
                          config={{
                            file: {
                              attributes: {
                                poster: classItem.thumbnail,
                                preload: 'auto',
                              }
                            }
                          }}
                        />
                      ) : (
                        <Image
                          src={classItem.thumbnail}
                          alt={classItem.title}
                          fill
                          className="object-cover"
                        />
                      )}
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                        {classItem.isLocked ? (
                          <Lock className="w-6 h-6 text-foreground" />
                        ) : (
                          <PlayCircle className="w-6 h-6 text-foreground" />
                        )}
                      </div>
                    </motion.div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-medium text-foreground truncate">
                          {classItem.title}
                        </h3>
                        {classItem.isCompleted && (
                          <CheckCircle2 className="w-4 h-4 text-primary flex-shrink-0" />
                        )}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <Clock className="w-4 h-4 text-muted-foreground" />
                        <p className="text-xs text-muted-foreground">
                          Duración: {classItem.duration}
                        </p>
                      </div>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${classItem.progress}%` }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            className="bg-primary h-2 rounded-full"
                          />
                        </div>
                        <span className="text-xs text-muted-foreground">{classItem.progress}%</span>
                      </div>
                      <motion.p 
                        className="text-xs text-primary mt-1 flex items-center gap-1"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 + index * 0.1 }}
                      >
                        <Award className="w-4 h-4" />
                        +{classItem.experience} XP
                      </motion.p>
                    </div>

                    {/* Status */}
                    <div className="flex-shrink-0">
                      {classItem.isLocked ? (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-muted text-muted-foreground">
                          Bloqueado
                        </span>
                      ) : classItem.isCompleted ? (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-primary/20 text-primary">
                          Completado
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-primary/20 text-primary">
                          Disponible
                        </span>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}    