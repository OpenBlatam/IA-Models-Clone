import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Copy, Share2, BookOpen, TrendingUp, Lightbulb, BarChart } from "lucide-react";
import { toast } from "sonner";
import { motion } from "framer-motion";
import { marked } from "marked";
import DOMPurify from "dompurify";
import slugify from "@/lib/utils";

interface BlogPost {
  title: string;
  excerpt: string;
  content: {
    sections: Array<{
      title: string;
      content: string;
    }>;
  };
  metadata: {
    keywords: string[];
    readingTime: string;
    seoDescription: string;
    slug: string;
    author: string;
    publishDate: string;
    category: string;
    tags: string[];
  };
}

interface BlogPostGeneratorProps {
  brandKit: any;
}

const blogTemplates = [
  {
    id: "guide",
    name: "Guía Completa",
    icon: BookOpen,
    description: "Tutorial detallado paso a paso",
    sections: [
      {
        title: "Introducción",
        type: "intro",
        content: "Introduce el tema y su importancia en el contexto actual. Explica por qué es relevante para la audiencia objetivo."
      },
      {
        title: "¿Qué es {topic}?",
        type: "definition",
        content: "Define el concepto de manera clara y concisa. Incluye ejemplos prácticos y casos de uso."
      },
      {
        title: "Beneficios de {topic}",
        type: "benefits",
        content: "Lista los principales beneficios y ventajas. Apoya cada punto con datos o ejemplos concretos."
      },
      {
        title: "Cómo implementar {topic}",
        type: "how-to",
        content: "Proporciona pasos detallados y accionables. Incluye consejos prácticos y mejores prácticas."
      },
      {
        title: "Mejores prácticas",
        type: "best-practices",
        content: "Comparte recomendaciones basadas en experiencia. Evita errores comunes y optimiza resultados."
      },
      {
        title: "Conclusión",
        type: "conclusion",
        content: "Resume los puntos clave y proporciona un llamado a la acción claro."
      }
    ],
    metadata: {
      keywords: ["guía", "tutorial", "aprender", "implementar"],
      readingTime: "15-20 min",
      category: "Tutoriales"
    }
  },
  {
    id: "analysis",
    name: "Análisis",
    icon: BarChart,
    description: "Análisis profundo de un tema",
    sections: [
      {
        title: "Contexto",
        type: "context",
        content: "Establece el marco de referencia y la situación actual del tema a analizar."
      },
      {
        title: "Análisis del mercado",
        type: "market-analysis",
        content: "Examina las tendencias, competidores y oportunidades en el mercado."
      },
      {
        title: "Tendencias actuales",
        type: "trends",
        content: "Identifica y analiza las tendencias más relevantes del momento."
      },
      {
        title: "Casos de estudio",
        type: "case-studies",
        content: "Presenta ejemplos reales y sus resultados. Extrae lecciones aprendidas."
      },
      {
        title: "Recomendaciones",
        type: "recommendations",
        content: "Ofrece sugerencias prácticas basadas en el análisis realizado."
      },
      {
        title: "Perspectivas futuras",
        type: "future",
        content: "Proyecta escenarios futuros y oportunidades de crecimiento."
      }
    ],
    metadata: {
      keywords: ["análisis", "mercado", "tendencias", "estudio"],
      readingTime: "20-25 min",
      category: "Análisis"
    }
  },
  {
    id: "tips",
    name: "Tips y Trucos",
    icon: Lightbulb,
    description: "Consejos prácticos y trucos",
    sections: [
      {
        title: "Introducción",
        type: "intro",
        content: "Presenta el tema y la importancia de los tips que se compartirán."
      },
      {
        title: "Tip #1: {tip1}",
        type: "tip",
        content: "Explica el primer tip con ejemplos prácticos y casos de uso."
      },
      {
        title: "Tip #2: {tip2}",
        type: "tip",
        content: "Detalla el segundo tip con pasos específicos y resultados esperados."
      },
      {
        title: "Tip #3: {tip3}",
        type: "tip",
        content: "Describe el tercer tip con consejos adicionales y mejores prácticas."
      },
      {
        title: "Consejos adicionales",
        type: "additional",
        content: "Comparte recomendaciones extra y trucos avanzados."
      },
      {
        title: "Resumen",
        type: "summary",
        content: "Recapitula los tips principales y su aplicación práctica."
      }
    ],
    metadata: {
      keywords: ["tips", "trucos", "consejos", "mejores prácticas"],
      readingTime: "10-15 min",
      category: "Consejos"
    }
  },
  {
    id: "trends",
    name: "Tendencias",
    icon: TrendingUp,
    description: "Análisis de tendencias actuales",
    sections: [
      {
        title: "Panorama actual",
        type: "overview",
        content: "Describe el estado actual del sector y las fuerzas que lo moldean."
      },
      {
        title: "Tendencia #1: {trend1}",
        type: "trend",
        content: "Analiza la primera tendencia, su impacto y casos de éxito."
      },
      {
        title: "Tendencia #2: {trend2}",
        type: "trend",
        content: "Examina la segunda tendencia, oportunidades y desafíos."
      },
      {
        title: "Tendencia #3: {trend3}",
        type: "trend",
        content: "Explora la tercera tendencia, innovaciones y aplicaciones prácticas."
      },
      {
        title: "Impacto en la industria",
        type: "impact",
        content: "Evalúa el impacto de las tendencias en diferentes sectores."
      },
      {
        title: "Conclusión",
        type: "conclusion",
        content: "Sintetiza las tendencias clave y su relevancia futura."
      }
    ],
    metadata: {
      keywords: ["tendencias", "innovación", "futuro", "tecnología"],
      readingTime: "12-18 min",
      category: "Tendencias"
    }
  }
];

export function BlogPostGenerator({ brandKit }: BlogPostGeneratorProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState(blogTemplates[0]);
  const [customTopic, setCustomTopic] = useState("");
  const [targetAudience, setTargetAudience] = useState("");
  const [generatedBlogPost, setGeneratedBlogPost] = useState<BlogPost | null>(null);

  const handleGenerateBlogPost = async () => {
    if (!customTopic || !targetAudience) {
      toast.error("Por favor completa todos los campos");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("/api/ads-ia", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          type: "blog",
          template: selectedTemplate.id,
          topic: customTopic,
          audience: targetAudience,
          brandKit,
          structure: selectedTemplate.sections
        }),
      });

      if (!response.ok) {
        throw new Error("Error al generar el post");
      }

      const data = await response.json();
      setGeneratedBlogPost(data);
      toast.success("¡Post generado con éxito!");
    } catch (error) {
      console.error("Error:", error);
      toast.error("Error al generar el post");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyToClipboard = () => {
    if (!generatedBlogPost) return;

    const content = `
# ${generatedBlogPost.title}

${generatedBlogPost.excerpt}

${generatedBlogPost.content.sections.map(section => `
## ${section.title}

${section.content}
`).join("\n\n")}

---
*Palabras clave: ${generatedBlogPost.metadata.keywords.join(", ")}*
*Tiempo de lectura: ${generatedBlogPost.metadata.readingTime}*
*Categoría: ${generatedBlogPost.metadata.category}*
    `.trim();

    navigator.clipboard.writeText(content);
    toast.success("Contenido copiado al portapapeles");
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-4">
          <div>
            <Label htmlFor="template">Template</Label>
            <Select
              value={selectedTemplate.id}
              onValueChange={(value) => {
                const template = blogTemplates.find(t => t.id === value);
                if (template) setSelectedTemplate(template);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Selecciona un template" />
              </SelectTrigger>
              <SelectContent>
                {blogTemplates.map((template) => (
                  <SelectItem key={template.id} value={template.id}>
                    <div className="flex items-center gap-2">
                      <template.icon className="h-4 w-4" />
                      <span>{template.name}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="topic">Tema del Post</Label>
            <Input
              id="topic"
              placeholder="Ej: Marketing Digital, SEO, Redes Sociales..."
              value={customTopic}
              onChange={(e) => setCustomTopic(e.target.value)}
            />
          </div>

          <div>
            <Label htmlFor="audience">Audiencia Objetivo</Label>
            <Textarea
              id="audience"
              placeholder="Describe a tu audiencia objetivo..."
              value={targetAudience}
              onChange={(e) => setTargetAudience(e.target.value)}
            />
          </div>

          <Button
            onClick={handleGenerateBlogPost}
            disabled={isLoading}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generando...
              </>
            ) : (
              "Generar Post de Blog"
            )}
          </Button>
        </div>

        <div className="space-y-4">
          {generatedBlogPost && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Vista Previa</span>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={handleCopyToClipboard}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="icon">
                      <Share2 className="h-4 w-4" />
                    </Button>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <h1>{generatedBlogPost.title}</h1>
                  <p className="text-muted-foreground">{generatedBlogPost.excerpt}</p>
                  <div className="space-y-6">
                    {generatedBlogPost.content.sections.map((section, index) => (
                      <div key={index}>
                        <h2>{section.title}</h2>
                        <div
                          dangerouslySetInnerHTML={{
                            __html: DOMPurify.sanitize(marked(section.content))
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 flex flex-wrap gap-2 text-sm text-muted-foreground">
                    <span>Palabras clave: {generatedBlogPost.metadata.keywords.join(", ")}</span>
                    <span>•</span>
                    <span>Tiempo de lectura: {generatedBlogPost.metadata.readingTime}</span>
                    <span>•</span>
                    <span>Categoría: {generatedBlogPost.metadata.category}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
} 