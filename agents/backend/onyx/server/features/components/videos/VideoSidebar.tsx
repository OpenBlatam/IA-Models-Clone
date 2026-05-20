"use client";
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { CheckCircle2, Lock, PlayCircle, Clock, Award } from "lucide-react";
import { AcademyClass } from "@/lib/types/academy";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useRouter } from "next/navigation";

interface VideoSidebarProps {
  open: boolean;
  onClose: () => void;
  currentIndex: number;
  onSelect: (index: number) => void;
  classes: AcademyClass[];
}

const VideoSidebar: React.FC<VideoSidebarProps> = ({
  open,
  onClose,
  currentIndex,
  onSelect,
  classes,
}) => {
  const router = useRouter();
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteSuccess, setInviteSuccess] = useState(false);

  // Simulación de mouse de invitado (puedes conectar lógica real después)
  const [guestCursor, setGuestCursor] = useState<{ x: number; y: number } | null>(null);

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className={cn(
        "fixed lg:relative top-0 right-0 h-full w-full lg:w-[400px] bg-background border-l border-border z-50 lg:z-0",
        open ? "translate-x-0" : "translate-x-full lg:translate-x-0"
      )}
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="sticky top-0 p-4 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10 flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <PlayCircle className="w-5 h-5" />
            Lista de Clases
          </h2>
          <button
            className="ml-auto px-3 py-1 rounded-lg bg-primary text-white font-semibold hover:bg-primary/80 transition-colors"
            onClick={() => setShowInviteModal(true)}
          >
            Invitar
          </button>
        </div>

        {/* Modal de invitación */}
        {showInviteModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
            <div className="bg-background rounded-xl shadow-2xl p-8 w-full max-w-sm relative">
              <button
                className="absolute top-2 right-2 text-muted-foreground hover:text-foreground"
                onClick={() => setShowInviteModal(false)}
              >
                ×
              </button>
              <h3 className="text-xl font-bold mb-4">Invitar a alguien</h3>
              <input
                type="email"
                placeholder="Correo electrónico"
                value={inviteEmail}
                onChange={e => setInviteEmail(e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-border mb-4 focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <button
                className="w-full py-2 rounded-lg bg-primary text-white font-semibold hover:bg-primary/80 transition-colors"
                onClick={() => {
                  setInviteSuccess(true);
                  setTimeout(() => {
                    setShowInviteModal(false);
                    setInviteSuccess(false);
                    setInviteEmail("");
                  }, 1500);
                }}
                disabled={!inviteEmail}
              >
                Enviar invitación
              </button>
              {inviteSuccess && (
                <div className="mt-4 text-green-600 font-semibold text-center">¡Invitación enviada!</div>
              )}
            </div>
          </div>
        )}

        {/* Mouse de invitado (simulado) */}
        {guestCursor && (
          <div
            className="fixed z-50 pointer-events-none"
            style={{
              left: guestCursor.x,
              top: guestCursor.y,
              width: 32,
              height: 32,
              transform: "translate(-50%, -50%)"
            }}
          >
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="16" fill="#6366f1" fillOpacity="0.7" />
              <path d="M10 10L22 16L10 22V10Z" fill="#fff" />
            </svg>
          </div>
        )}

        {/* Classes List */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence>
            {classes.map((classItem, index) => (
              <motion.div
                key={classItem.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ delay: index * 0.05 }}
                className="border-b border-border/50 last:border-0"
              >
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        className={cn(
                          "w-full h-auto p-4 flex items-start gap-4 text-left hover:bg-accent/50 transition-all duration-200",
                          currentIndex === index && "bg-accent/80"
                        )}
                        onClick={() => {
                          onSelect(index);
                          router.push(`/dashboard/videos?courseId=${classItem.academyId}&classId=${classItem.id}`);
                        }}
                      >
                        <div className="relative flex-shrink-0 group">
                          <div className="w-[180px] h-[100px] rounded-lg overflow-hidden">
                            <img
                              src={classItem.thumbnail}
                              alt={classItem.title}
                              className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                            />
                          </div>
                          <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                            {classItem.isLocked ? (
                              <Lock className="w-8 h-8 text-white" />
                            ) : classItem.isCompleted ? (
                              <CheckCircle2 className="w-8 h-8 text-green-500" />
                            ) : (
                              <PlayCircle className="w-8 h-8 text-white" />
                            )}
                          </div>
                          <div className="absolute bottom-1 right-1 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded">
                            {classItem.duration}
                          </div>
                        </div>
                        <div className="flex-1 min-w-0 space-y-2">
                          <div>
                            <h3 className="font-medium text-sm line-clamp-2 mb-1 group-hover:text-primary transition-colors">
                              {classItem.title}
                            </h3>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              <span>{classItem.duration}</span>
                              {classItem.experience > 0 && (
                                <>
                                  <span>•</span>
                                  <div className="flex items-center gap-1">
                                    <Award className="w-3 h-3 text-yellow-500" />
                                    <span>{classItem.experience} XP</span>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>
                          {classItem.progress && classItem.progress > 0 && (
                            <div className="space-y-1">
                              <Progress value={classItem.progress} className="h-1" />
                              <div className="flex justify-between text-xs text-muted-foreground">
                                <span>Progreso</span>
                                <span>{classItem.progress}%</span>
                              </div>
                            </div>
                          )}
                        </div>
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="left" className="max-w-[300px]">
                      <p className="text-sm">{classItem.title}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {classItem.isLocked ? "Clase bloqueada" : 
                         classItem.isCompleted ? "Clase completada" : 
                         "Disponible para ver"}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
};

export default VideoSidebar;    