"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Play, Clock, Users, Award } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Academy, AcademyClass } from "@/lib/types/academy";
import AcademyClassesModal from "./AcademyClassesModal";

interface AcademyCarouselProps {
  academies: Academy[];
  onSelectAcademy: (academyId: string) => void;
  title: string;
  subtitle?: string;
  showProgress?: boolean;
}

export default function AcademyCarousel({
  academies,
  onSelectAcademy,
  title,
  subtitle,
  showProgress = false,
}: AcademyCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showClasses, setShowClasses] = useState(false);
  const [selectedAcademy, setSelectedAcademy] = useState<Academy | null>(null);

  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : academies.length - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev < academies.length - 1 ? prev + 1 : 0));
  };

  const handleAcademyClick = (academy: Academy) => {
    setSelectedAcademy(academy);
    setShowClasses(true);
  };

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
          {subtitle && (
            <p className="text-muted-foreground">{subtitle}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={handlePrevious}
            className="rounded-full"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={handleNext}
            className="rounded-full"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Carousel */}
      <div className="relative">
        <motion.div
          key={currentIndex}
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -100 }}
          transition={{ duration: 0.3 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {academies.slice(currentIndex, currentIndex + 3).map((academy) => (
            <Card
              key={academy.id}
              className="group cursor-pointer transition-all duration-300 hover:shadow-lg"
              onClick={() => handleAcademyClick(academy)}
            >
              <CardHeader className="relative aspect-video p-0">
                <img
                  src={academy.thumbnail}
                  alt={academy.name}
                  className="w-full h-full object-cover rounded-t-lg"
                />
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <Button variant="secondary" size="lg">
                    <Play className="w-6 h-6 mr-2" />
                    Ver Academia
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Badge variant="secondary">{academy.level}</Badge>
                    <Badge variant="outline">{academy.category}</Badge>
                  </div>
                  <h3 className="text-xl font-semibold line-clamp-1">
                    {academy.name}
                  </h3>
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {academy.description}
                  </p>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{academy.totalDuration}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Users className="w-4 h-4" />
                      <span>{academy.totalClasses} clases</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Award className="w-4 h-4" />
                      <span>{academy.experience} XP</span>
                    </div>
                  </div>
                </div>
              </CardContent>
              {showProgress && (
                <CardFooter className="p-4 pt-0">
                  <div className="w-full space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>Progreso</span>
                      <span>{academy.totalProgress}%</span>
                    </div>
                    <Progress value={academy.totalProgress} />
                  </div>
                </CardFooter>
              )}
            </Card>
          ))}
        </motion.div>
      </div>

      {/* Classes Modal */}
      {selectedAcademy && (
        <AcademyClassesModal
          open={showClasses}
          onClose={() => setShowClasses(false)}
          academy={selectedAcademy}
          onSelectClass={(classId) => {
            onSelectAcademy(classId);
            setShowClasses(false);
          }}
        />
      )}
    </div>
  );
}     