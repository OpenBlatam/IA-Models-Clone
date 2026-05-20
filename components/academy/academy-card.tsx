"use client";

import { Academy } from "@/lib/types/academy";
import Link from "next/link";
import { motion } from "framer-motion";
import { Thumbnail } from "./thumbnail";
import { InstructorInfo } from "./instructor-info";
import { CourseMeta } from "./course-meta";
import { Star, ArrowRight, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useRouter } from "next/navigation";

interface AcademyCardProps {
  academy: Academy;
  onSelect?: (academy: Academy) => void;
  variant?: "grid" | "list";
}

export function AcademyCard({ academy, onSelect, variant = "grid" }: AcademyCardProps) {
  const router = useRouter();
  const handleSelect = (e: React.MouseEvent) => {
    e.preventDefault();
    if (onSelect) onSelect(academy);
    // Navegación directa a la página de videos de la academia
    router.push(`/dashboard/videos?courseId=${academy.id}`);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.04, boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.37)" }}
      transition={{ duration: 0.25, type: "spring" }}
      className="group bg-gradient-to-br from-[#232326] to-[#18181b] rounded-2xl p-5 shadow-xl hover:shadow-2xl transition-shadow"
    >
      <div className="relative w-full aspect-video rounded-xl overflow-hidden mb-4">
        <img
          src={academy.thumbnail}
          alt={academy.name}
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
        <div className="absolute bottom-3 right-3 bg-[#232326]/80 backdrop-blur-md rounded-lg px-3 py-1 flex items-center gap-1 text-yellow-400 font-bold text-base shadow-lg z-10">
          <Star size={18} /> {academy.rating?.toFixed(1) ?? "4.8"}
        </div>
        <div className="absolute inset-0 flex items-center justify-center gap-4 bg-gradient-to-br from-black/60 to-[#232326]/70 backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-20">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="lg"
                  className="text-base font-semibold gap-2 shadow-lg"
                  onClick={handleSelect}
                >
                  Ir al curso <ArrowRight className="w-5 h-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Ver detalles del curso</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="lg"
                  variant="secondary"
                  className="text-base font-semibold gap-2 shadow-lg"
                  onClick={handleSelect}
                >
                  Agregar <Plus className="w-5 h-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Agregar a favoritos</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
      <div className="flex items-center gap-4 mb-2">
        <img
          src={`https://ui-avatars.com/api/?name=${encodeURIComponent(academy.name)}&background=8b5cf6&color=fff&size=80`}
          alt={academy.name}
          className="w-12 h-12 rounded-full border-4 border-primary bg-white shadow-lg"
        />
        <span className="text-xl font-bold text-white line-clamp-2">{academy.name}</span>
      </div>
      <div className="text-base text-[#b3b3b3] font-medium">Por {academy.instructor}</div>
    </motion.div>
  );
}    