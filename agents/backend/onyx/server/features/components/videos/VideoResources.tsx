"use client";
import { FileText, BookOpen, Download, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { LoadingState } from "@/components/ui/loading-state";
import { useVideoData } from "@/lib/hooks/useVideoData";

interface VideoResourcesProps {
  videoId: string;
  courseId: string;
}

export function VideoResources({ videoId, courseId }: VideoResourcesProps) {
  const { data, isLoading, error } = useVideoData(videoId, courseId);

  if (isLoading) {
    return <LoadingState text="Cargando recursos..." />;
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-destructive">{error}</p>
      </div>
    );
  }

  const files = data?.resources.filter(r => r.type === "file") || [];
  const readings = data?.resources.filter(r => r.type === "reading") || [];

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-foreground">Recursos</h2>

      {/* Files Section */}
      {files.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Archivos
          </h3>
          <div className="grid gap-4">
            {files.map((file) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-between p-4 rounded-lg bg-muted border border-border"
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-muted-foreground" />
                  <span className="text-foreground">{file.title}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => window.open(file.url, "_blank")}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Descargar
                </Button>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Readings Section */}
      {readings.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Lecturas
          </h3>
          <div className="grid gap-4">
            {readings.map((reading) => (
              <motion.div
                key={reading.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-between p-4 rounded-lg bg-muted border border-border"
              >
                <div className="flex items-center gap-3">
                  <BookOpen className="w-5 h-5 text-muted-foreground" />
                  <span className="text-foreground">{reading.title}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => window.open(reading.url, "_blank")}
                >
                  <ExternalLink className="w-4 h-4 mr-2" />
                  Abrir
                </Button>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {files.length === 0 && readings.length === 0 && (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No hay recursos disponibles para esta clase.</p>
        </div>
      )}
    </div>
  );
}    