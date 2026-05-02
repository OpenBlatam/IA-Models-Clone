"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, AlertCircle, BookOpen } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { BrandKitWizardModal } from "@/components/BrandKit/BrandKitWizardModal";
import { parseBrandKit } from "@/utils/parseBrandKit";
import { BrandKit as BrandKitComponent } from "@/components/BrandKit/BrandKit";
import { TemplateCarousel, templateCategories, TemplateDetailModal } from "@/components/BrandKit/TemplateSelector";
import { AudienceModal } from "@/components/BrandKit/AudienceModal";
import type { BrandKitData } from "@/components/BrandKit/BrandKit";
import { motion, AnimatePresence } from "framer-motion";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Sparkles, Send, Wand2, Palette, Rocket, Crown, Star, Gem } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import BrandKitDisplay from "@/components/BrandKit/BrandKitDisplay";
import HeroSection from "@/components/ui/HeroSection";
import UrlInputBox from "@/components/ui/UrlInputBox";
import { FacebookPostGenerator } from "@/components/FacebookPost/FacebookPostGenerator";
import { BlogPostGenerator } from "@/components/BlogPost/BlogPostGenerator";

function mapBrandKitDataToDisplay(data) {
  return {
    brand: {
      name: "Brand Name",
      logo: "/logo-placeholder.png",
    },
    voice: {
      purpose: data.fonts?.title || "",
      audience: data.fonts?.subtitle || "",
      tone: [],
      emotions: [],
    },
    materials: [],
    media: [],
  };
}

export default function AdsIaPage() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [ads, setAds] = useState<string[]>([]);
  const [brandKit, setBrandKit] = useState("");
  const [error, setError] = useState<{ message: string; details?: string } | null>(null);
  const [activeTab, setActiveTab] = useState("ads");
  const [wizardOpen, setWizardOpen] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [brandKitData, setBrandKitData] = useState<BrandKitData | null>(null);
  const [selectedTemplateForAudience, setSelectedTemplateForAudience] = useState<{ id: number; title: string; color: string; image: string } | null>(null);
  const [audienceModalOpen, setAudienceModalOpen] = useState(false);
  const [audienceLoading, setAudienceLoading] = useState(false);
  const [generatedContent, setGeneratedContent] = useState("");
  const [selectedTemplateDetail, setSelectedTemplateDetail] = useState(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [customText, setCustomText] = useState("");

  const handleExtract = async (type: "ads" | "brand-kit") => {
    setLoading(true);
    setError(null);
    setAds([]);
    setBrandKit("");
    
    try {
      console.log('Making request to /api/ads-ia with:', { url, type });
      const res = await fetch("/api/ads-ia", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, type }),
      });
      
      console.log('Response status:', res.status);
      const contentType = res.headers.get('content-type');
      console.log('Response content type:', contentType);
      
      if (!contentType || !contentType.includes('application/json')) {
        const text = await res.text();
        console.error('Non-JSON response:', text);
        throw new Error('Invalid response format from server');
      }
      
      const data = await res.json();
      console.log('Response data:', data);
      
      if (!res.ok) {
        throw new Error(data.error || "Error al procesar la solicitud");
      }
      
      if (data.error && !data.fallback) {
        throw new Error(data.error);
      }
      
      if (type === "ads") {
        setAds(data.ads || []);
      } else {
        setBrandKit(data.brandKit || "");
        if (data.brandKit) {
          setWizardOpen(true);
        }
      }
    } catch (e: any) {
      console.error('Error in handleExtract:', e);
      setError({
        message: e.message || "Error al procesar la solicitud",
        details: e.details || e.message
      });
    } finally {
      setLoading(false);
    }
  };

  // Streaming handler for ads
  const handleExtractStream = async () => {
    setLoading(true);
    setError(null);
    setAds([]);
    try {
      const payload: any = { url, type: "ads" };
      if (customText.trim()) {
        payload.website_content = customText;
      }
      const res = await fetch("/api/ads-ia/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.body) throw new Error("No stream body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = '';
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        let lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (line.trim()) setAds(prev => [...prev, line]);
        }
      }
      if (buffer.trim()) setAds(prev => [...prev, buffer]);
    } catch (e: any) {
      setError({ message: e.message || "Error al procesar la solicitud" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (brandKit) {
      setBrandKitData(parseBrandKit(brandKit));
    }
  }, [brandKit]);

  const handleGenerateContent = async (audience: string) => {
    setAudienceLoading(true);
    setGeneratedContent("");
    try {
      const prompt = `Genera un post para ${audience} usando el template "${selectedTemplateForAudience?.title}".`;
      const res = await fetch("/api/ads-ia", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          url, 
          type: "ads", 
          prompt,
          brandKit: brandKitData // Pass brand kit data for better context
        }),
      });
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || "Error al generar contenido");
      }
      
      setGeneratedContent(data.ads?.join("\n\n") || "No se pudo generar contenido.");
    } catch (e: any) {
      setGeneratedContent("Error al generar contenido: " + (e.message || "Error desconocido"));
    } finally {
      setAudienceLoading(false);
    }
  };

  const handleGenerateFacebookPost = async () => {
    setLoading(true);
    setError(null);
    try {
      const prompt = `Genera un post para Facebook con el siguiente formato:
      - Título atractivo
      - Contenido principal (2-3 párrafos)
      - URL de imagen sugerida
      - Call to action
      - Hashtags relevantes
      
      Usando el brand kit: ${JSON.stringify(brandKitData)}`;

      const res = await fetch("/api/ads-ia", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          url, 
          type: "facebook-post", 
          prompt,
          brandKit: brandKitData
        }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || "Error al generar el post de Facebook");
      }
      
      // Parse the response and update the Facebook post state
      const postData = data.facebookPost || {};
      setGeneratedContent(postData.content || "");
    } catch (e: any) {
      setError({
        message: e.message || "Error al generar el post de Facebook",
        details: e.details || e.message
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <HeroSection
        title={
          <span>
            Herramientas de{" "}
            <motion.span
              className="bg-gradient-to-r from-[#7b61ff] via-[#fbbf24] to-[#22c55e] bg-clip-text text-transparent font-extrabold tracking-tight drop-shadow-lg"
              initial={{ backgroundPosition: "0% 50%" }}
              animate={{ backgroundPosition: "100% 50%" }}
              transition={{
                repeat: Infinity,
                repeatType: "reverse",
                duration: 4,
                ease: "linear"
              }}
              style={{
                backgroundSize: "200% 200%",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent"
              }}
            >
              Marketing
            </motion.span>
          </span>
        }
        subtitle="Genera anuncios y brand kits automáticamente a partir de cualquier URL o texto."
        backgroundClass="bg-black"
        textClass="text-white"
        linearGradientBackground={true}
        multicolorBackground={false}
      >
        <UrlInputBox
          value={url}
          onChange={setUrl}
          loading={loading}
          onGenerate={() => handleExtract(activeTab as "ads" | "brand-kit")}
          onTryExample={() => setUrl("https://www.apple.com")}
          onUseText={() => setUrl("Ejemplo de texto para anuncio")}
          extraClass=""
        />
        <textarea
          placeholder="Pega aquí tu texto para generar anuncios (opcional)"
          value={customText}
          onChange={e => setCustomText(e.target.value)}
          className="w-full mt-2 p-2 rounded border border-gray-300 text-black"
          rows={3}
        />
        <Button onClick={handleExtractStream} disabled={loading} className="ml-4 mt-2">
          Generar anuncios (stream)
        </Button>
      </HeroSection>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-5xl mx-auto py-12 space-y-8 px-4 relative"
      >
        {/* Decorative elements */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.1 }}
          transition={{ delay: 1 }}
          className="absolute inset-0 pointer-events-none"
        >
          <div className="absolute top-0 left-1/4 w-64 h-64 bg-primary/20 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-primary/20 rounded-full blur-3xl" />
        </motion.div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="ads">
              <div className="flex items-center gap-2">
                <Rocket className="w-4 h-4" />
                Anuncios
              </div>
            </TabsTrigger>
            <TabsTrigger value="brand-kit">
              <div className="flex items-center gap-2">
                <Palette className="w-4 h-4" />
                Brand Kit
              </div>
            </TabsTrigger>
            <TabsTrigger value="facebook">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                </svg>
                Facebook
              </div>
            </TabsTrigger>
            <TabsTrigger value="blog">
              <div className="flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                Blog
              </div>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="ads" className="space-y-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card className="border-2 hover:border-primary/50 transition-all duration-200 bg-background/50 backdrop-blur-sm shadow-xl shadow-primary/5">
                <CardHeader>
                  <CardTitle className="text-2xl flex items-center gap-2">
                    <Rocket className="w-6 h-6 text-primary" />
                    Generador de Anuncios
                  </CardTitle>
                  <CardDescription className="text-lg">
                    Genera anuncios optimizados para redes sociales a partir de cualquier web.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <Alert variant="destructive" className="mt-4">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>
                          {error.message}
                          {error.details && error.details !== error.message && (
                            <p className="mt-2 text-sm opacity-80">{error.details}</p>
                          )}
                        </AlertDescription>
                      </Alert>
                    </motion.div>
                  )}
                  
                  {ads.length > 0 && (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-6 space-y-4"
                    >
                      <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-primary" />
                        Anuncios Generados:
                      </h2>
                      <ul className="space-y-3">
                        {ads.map((ad, i) => (
                          <motion.li 
                            key={i}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="bg-muted/50 backdrop-blur-sm rounded-xl p-4 border border-border hover:border-primary/50 transition-all duration-200 group shadow-lg shadow-primary/5"
                          >
                            <div className="flex items-start gap-3">
                              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-1 ring-2 ring-primary/20">
                                <span className="text-primary font-medium">{i + 1}</span>
                              </div>
                              <p className="text-base leading-relaxed">{ad}</p>
                            </div>
                          </motion.li>
                        ))}
                      </ul>
                    </motion.div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          <TabsContent value="brand-kit" className="space-y-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card className="border-2 hover:border-primary/50 transition-all duration-200 bg-background/50 backdrop-blur-sm shadow-xl shadow-primary/5">
                <CardHeader>
                  <CardTitle className="text-2xl flex items-center gap-2">
                    <Palette className="w-6 h-6 text-primary" />
                    Generador de Brand Kit
                  </CardTitle>
                  <CardDescription className="text-lg">
                    Extrae y genera un brand kit completo a partir de cualquier web.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <Alert variant="destructive" className="mt-4">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>
                          {error.message}
                          {error.details && error.details !== error.message && (
                            <p className="mt-2 text-sm opacity-80">{error.details}</p>
                          )}
                        </AlertDescription>
                      </Alert>
                    </motion.div>
                  )}
                  
                  {brandKit && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-6"
                    >
                      <div className="mt-6 space-y-4">
                        <h2 className="text-xl font-semibold flex items-center gap-2">
                          <Sparkles className="w-5 h-5 text-primary" />
                          Brand Kit Generado:
                        </h2>
                        <div className="bg-muted/50 backdrop-blur-sm rounded-xl p-4 border border-border hover:border-primary/50 transition-all duration-200 shadow-lg shadow-primary/5">
                          {brandKit}
                        </div>
                      </div>
                      {brandKitData && (
                        <div className="mt-10">
                          <BrandKitDisplay
                            {...mapBrandKitDataToDisplay(brandKitData)}
                            onCreate={() => {}}
                          />
                        </div>
                      )}
                    </motion.div>
                  )}
                  <div className="space-y-12 mt-8">
                    {templateCategories.map(cat => (
                      <TemplateCarousel
                        key={cat.title}
                        title={cat.title}
                        templates={cat.templates}
                        onCardClick={tpl => {
                          setSelectedTemplateDetail(tpl);
                          setDetailModalOpen(true);
                        }}
                      />
                    ))}
                  </div>
                  <AudienceModal
                    open={audienceModalOpen}
                    onClose={() => {
                      setAudienceModalOpen(false);
                      setGeneratedContent("");
                    }}
                    onRegenerate={() => handleGenerateContent("")}
                    onNext={(audience, brandKitId, language) => {
                      handleGenerateContent(audience);
                    }}
                    suggestion={selectedTemplateForAudience?.title || ""}
                    loading={audienceLoading}
                    brandKitId={1}
                    language="es"
                  />
                  <TemplateDetailModal
                    open={detailModalOpen}
                    onClose={() => setDetailModalOpen(false)}
                    template={selectedTemplateDetail}
                    onUseTemplate={tpl => {
                      setSelectedTemplateForAudience(tpl);
                      setAudienceModalOpen(true);
                    }}
                  />
                  <BrandKitWizardModal
                    open={wizardOpen}
                    onClose={() => setWizardOpen(false)}
                    onGenerate={tpl => {
                      setSelectedTemplate(tpl);
                      setWizardOpen(false);
                    }}
                    brandKit={brandKit}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          <TabsContent value="facebook" className="space-y-4">
            <FacebookPostGenerator
              url={url || ""}
              brandKitData={brandKitData}
              onError={setError}
            />
          </TabsContent>

          <TabsContent value="blog" className="space-y-4">
            <BlogPostGenerator
              url={url || ""}
              brandKitData={brandKitData}
              onError={setError}
            />
          </TabsContent>
        </Tabs>
      </motion.div>
    </>
  );
} 