import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Crown, Sparkles, Zap, Star, Bot } from "lucide-react";

export function UpgradeCard() {
  return (
    <Card className="relative overflow-hidden border-2 border-purple-400/20 bg-gradient-to-br from-purple-500/5 to-blue-500/5 backdrop-blur-sm">
      <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-blue-500/10 opacity-50" />
      <div className="absolute -right-12 -top-12 h-24 w-24 rotate-12 transform rounded-full bg-purple-500/20 blur-2xl" />
      <div className="absolute -left-12 -bottom-12 h-24 w-24 rotate-12 transform rounded-full bg-blue-500/20 blur-2xl" />
      
      <CardHeader className="relative">
        <div className="flex items-center gap-2 mb-4">
          <Bot className="h-6 w-6 text-blue-500 animate-pulse" />
          <CardTitle className="text-lg bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Asistente IA Personal
          </CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Crown className="h-5 w-5 text-yellow-500" />
          <CardTitle className="bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            ¡Desbloquea Todo con Premium! 🚀
          </CardTitle>
        </div>
        <CardDescription className="text-base">
          Accede a todas las características premium:
        </CardDescription>
      </CardHeader>
      
      <CardContent className="relative space-y-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Sparkles className="h-4 w-4 text-yellow-500" />
            <span>Acceso a todas las lecciones</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Zap className="h-4 w-4 text-yellow-500" />
            <span>Soporte VIP prioritario</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Star className="h-4 w-4 text-yellow-500" />
            <span>Certificaciones exclusivas</span>
          </div>
        </div>
        
        <Button 
          size="sm" 
          className="w-full bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
        >
          Actualizar Ahora - $8.99/mes
        </Button>
      </CardContent>
    </Card>
  );
}
