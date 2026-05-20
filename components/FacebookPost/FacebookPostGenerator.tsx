import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Sparkles, Copy, Share2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import type { BrandKitData } from "@/components/BrandKit/BrandKit";
import { toast } from "sonner";

interface FacebookPost {
  title: string;
  content: string;
  imageUrl: string;
  callToAction: string;
  hashtags: string[];
  engagement: {
    likes: number;
    comments: number;
    shares: number;
  };
}

interface FacebookPostGeneratorProps {
  url: string;
  brandKitData: BrandKitData | null;
  onError: (error: { message: string; details?: string }) => void;
}

const POST_TEMPLATES = [
  {
    name: "Producto Nuevo",
    prompt: "Anuncia un nuevo producto con emoción y exclusividad",
    emojis: ["✨", "🎉", "🔥", "💫"]
  },
  {
    name: "Oferta Especial",
    prompt: "Crea una oferta irresistible con urgencia y valor",
    emojis: ["⚡", "💰", "🎁", "⏰"]
  },
  {
    name: "Contenido Educativo",
    prompt: "Comparte conocimiento valioso de forma atractiva",
    emojis: ["📚", "🎯", "💡", "🎓"]
  },
  {
    name: "Historia de Éxito",
    prompt: "Cuenta una historia inspiradora de un cliente",
    emojis: ["🌟", "💪", "🏆", "🙌"]
  }
];

export function FacebookPostGenerator({ url, brandKitData, onError }: FacebookPostGeneratorProps) {
  const [loading, setLoading] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState(POST_TEMPLATES[0]);
  const [facebookPost, setFacebookPost] = useState<FacebookPost>({
    title: "",
    content: "",
    imageUrl: "",
    callToAction: "",
    hashtags: [],
    engagement: {
      likes: 0,
      comments: 0,
      shares: 0
    }
  });

  const handleGenerateFacebookPost = async () => {
    setLoading(true);
    try {
      const prompt = `Genera un post para Facebook optimizado para engagement con el siguiente formato:
      - Título impactante y emocional (máximo 60 caracteres)
      - Contenido principal (2 párrafos cortos) usando emojis estratégicos: ${selectedTemplate.emojis.join(" ")}
      - URL de imagen sugerida que sea viral y relevante
      - Call to action persuasivo y claro
      - Hashtags populares y relevantes (máximo 5)
      
      Template: ${selectedTemplate.prompt}
      Brand Kit: ${JSON.stringify(brandKitData)}
      
      El post debe ser:
      - Emocional y personal
      - Fácil de leer
      - Optimizado para engagement
      - Con elementos virales
      - Adaptado al algoritmo de Facebook`;

      const res = await fetch("/api/ads-ia", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          url, 
          type: "facebook-post", 
          prompt,
          brandKit: brandKitData,
          template: selectedTemplate.name
        }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || "Error al generar el post de Facebook");
      }
      
      const postData = data.facebookPost || {};
      setFacebookPost({
        title: postData.title || "",
        content: postData.content || "",
        imageUrl: postData.imageUrl || "",
        callToAction: postData.callToAction || "",
        hashtags: postData.hashtags || [],
        engagement: {
          likes: Math.floor(Math.random() * 1000) + 100,
          comments: Math.floor(Math.random() * 100) + 10,
          shares: Math.floor(Math.random() * 50) + 5
        }
      });
    } catch (e: any) {
      onError({
        message: e.message || "Error al generar el post de Facebook",
        details: e.details || e.message
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    const text = `${facebookPost.title}\n\n${facebookPost.content}\n\n${facebookPost.callToAction}\n\n${facebookPost.hashtags.join(" ")}`;
    await navigator.clipboard.writeText(text);
    toast.success("Post copiado al portapapeles");
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Generador de Posts para Facebook</CardTitle>
        <CardDescription>
          Crea posts optimizados para engagement en Facebook
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {POST_TEMPLATES.map((template) => (
              <Button
                key={template.name}
                variant={selectedTemplate.name === template.name ? "default" : "outline"}
                onClick={() => setSelectedTemplate(template)}
                className="flex flex-col items-center gap-1 h-auto py-2"
              >
                <span className="text-lg">{template.emojis[0]}</span>
                <span className="text-xs">{template.name}</span>
              </Button>
            ))}
          </div>

          <Button 
            onClick={handleGenerateFacebookPost} 
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generando post...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generar Post de Facebook
              </>
            )}
          </Button>

          {facebookPost.title && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="space-y-4 p-4 border rounded-lg bg-white shadow-lg"
            >
              <div className="flex justify-between items-start">
                <h3 className="text-xl font-bold text-gray-900">{facebookPost.title}</h3>
                <div className="flex gap-2">
                  <Button variant="ghost" size="icon" onClick={copyToClipboard}>
                    <Copy className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="icon">
                    <Share2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <p className="whitespace-pre-wrap text-gray-700">{facebookPost.content}</p>
              
              {facebookPost.imageUrl && (
                <div className="relative aspect-video w-full overflow-hidden rounded-lg shadow-md">
                  <img 
                    src={facebookPost.imageUrl} 
                    alt="Post preview" 
                    className="object-cover w-full h-full"
                  />
                </div>
              )}
              
              <div className="flex flex-wrap gap-2">
                {facebookPost.hashtags.map((tag, index) => (
                  <Badge 
                    key={index} 
                    variant="secondary"
                    className="bg-blue-100 text-blue-800 hover:bg-blue-200"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
              
              <div className="text-sm text-gray-500 font-medium">
                Call to Action: {facebookPost.callToAction}
              </div>

              <div className="flex items-center gap-4 pt-2 border-t">
                <div className="flex items-center gap-1">
                  <span className="text-lg">❤️</span>
                  <span className="text-sm text-gray-600">{facebookPost.engagement.likes}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-lg">💬</span>
                  <span className="text-sm text-gray-600">{facebookPost.engagement.comments}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-lg">🔄</span>
                  <span className="text-sm text-gray-600">{facebookPost.engagement.shares}</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 