"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Save, FileText, Download, Sparkles, Edit, Trash2, Send, Copy,
  Bold, Italic, Underline, AlignLeft, AlignCenter, AlignRight,
  List, ListOrdered, Image, Table, Link, Code, Type, FileUp,
  ChevronDown, Settings, HelpCircle, File, Home, Layout, Eye,
  MessageSquare, Paintbrush, Search, BookOpen, FileCheck
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface OnlyOfficeEditorProps {
  documentUrl?: string;
  documentTitle?: string;
  documentType?: "text" | "spreadsheet" | "presentation";
  onSave?: (url: string) => void;
}

declare global {
  interface Window {
    DocsAPI: any;
  }
}

export default function OnlyOfficeEditor({
  documentUrl = "https://example.com/url-to-callback",
  documentTitle = "Nuevo Documento",
  documentType = "text",
  onSave,
}: OnlyOfficeEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [aiError, setAiError] = useState("");
  const [aiText, setAiText] = useState("");
  const [docEditor, setDocEditor] = useState<any>(null);
  const [showAIPanel, setShowAIPanel] = useState(false);
  const [activeTab, setActiveTab] = useState("home");

  useEffect(() => {
    if (!editorRef.current || typeof window === 'undefined') return;

    const script = document.createElement("script");
    script.src = "https://documentserver.your-domain.com/web-apps/apps/api/documents/api.js";
    script.async = true;
    document.body.appendChild(script);

    script.onload = () => {
      if (typeof window.DocsAPI === "undefined") {
        console.error("ONLYOFFICE Docs API failed to load");
        return;
      }

      const uniqueKey = `${Date.now()}-${Math.random().toString(36).substring(7)}`;

      const instance = new window.DocsAPI.DocEditor(editorRef.current, {
        width: "100%",
        height: "100%",
        documentType: documentType,
        document: {
          title: documentTitle,
          url: documentUrl,
          fileType: documentType === "text" ? "docx" : documentType === "spreadsheet" ? "xlsx" : "pptx",
          key: uniqueKey,
          permissions: {
            edit: true,
            download: true,
            print: true,
            fillForms: true,
            comment: true,
            copy: true,
            modifyContentControl: true,
            modifyFilter: true,
          },
        },
        editorConfig: {
          mode: "edit",
          lang: "es",
          callbackUrl: documentUrl,
          customization: {
            chat: false,
            comments: true,
            feedback: false,
            forcesave: true,
            toolbar: {
              showReviewTab: true,
              showPlugins: true,
              showComments: true,
            },
            about: false,
            help: false,
            plugins: false,
            macros: false,
            macrosMode: false,
            review: {
              showReviewChanges: true,
              showReviewPanel: true,
            },
            layout: {
              showStatusBar: true,
              showToolbar: true,
              showRuler: true,
              showScrollbar: true,
              showTabs: true,
              showTabsBar: true,
              showTabsPanel: true,
              showTabsPanelButton: true,
              showTabsPanelButtonText: true,
              showTabsPanelButtonIcon: true,
              showTabsPanelButtonTooltip: true,
              showTabsPanelButtonShortcut: true,
              showTabsPanelButtonShortcutText: true,
              showTabsPanelButtonShortcutIcon: true,
              showTabsPanelButtonShortcutTooltip: true,
            },
          },
        },
        events: {
          onAppReady: () => {
          },
          onDocumentStateChange: (event: any) => {
          },
          onError: (event: any) => {
            console.error("Editor error:", event);
          },
        },
      });

      setDocEditor(instance);
    };

    return () => {
      if (typeof document !== 'undefined' && document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, [documentUrl, documentTitle, documentType]);

  const handleAIGenerate = async () => {
    setLoading(true);
    setAiError("");
    setAiText("");
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error("Error al generar texto IA");
      const data = await res.json();
      setAiText(data.text);
      setPrompt("");
    } catch (err: any) {
      setAiError(err.message || "Error inesperado");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (aiText && typeof navigator !== 'undefined' && navigator.clipboard) {
      await navigator.clipboard.writeText(aiText);
    }
  };

  const ToolbarButton = ({ icon: Icon, tooltip, onClick, active }: { icon: any; tooltip: string; onClick?: () => void; active?: boolean }) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant={active ? "secondary" : "ghost"}
            size="icon"
            className={`h-8 w-8 ${active ? 'bg-[#2b579a]/10 text-[#2b579a]' : 'text-gray-700 hover:bg-[#2b579a]/10 hover:text-[#2b579a]'}`}
            onClick={onClick}
          >
            <Icon className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="bg-[#2b579a] text-white border-none">
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  return (
    <div className="relative w-full min-h-[calc(100vh-120px)] flex flex-col bg-[#f3f3f3]">
      {/* Barra de título */}
      <motion.div 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="w-full bg-[#2b579a] text-white px-4 py-2.5 flex items-center justify-between shadow-md"
      >
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            <span className="font-medium text-base">{documentTitle}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ToolbarButton icon={Save} tooltip="Guardar" />
          <ToolbarButton icon={Download} tooltip="Descargar" />
          <ToolbarButton icon={Settings} tooltip="Configuración" />
          <ToolbarButton icon={HelpCircle} tooltip="Ayuda" />
        </div>
      </motion.div>

      {/* Cinta de opciones */}
      <div className="w-full bg-white border-b border-gray-200 shadow-sm">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="h-11 px-2 bg-[#f3f3f3] border-b border-gray-200">
            <TabsTrigger 
              value="file" 
              className="h-9 px-4 data-[state=active]:bg-white data-[state=active]:text-[#2b579a] data-[state=active]:shadow-sm font-medium"
            >
              <File className="w-4 h-4 mr-2" />
              Archivo
            </TabsTrigger>
            <TabsTrigger 
              value="home" 
              className="h-9 px-4 data-[state=active]:bg-white data-[state=active]:text-[#2b579a] data-[state=active]:shadow-sm font-medium"
            >
              <Home className="w-4 h-4 mr-2" />
              Inicio
            </TabsTrigger>
            <TabsTrigger 
              value="insert" 
              className="h-9 px-4 data-[state=active]:bg-white data-[state=active]:text-[#2b579a] data-[state=active]:shadow-sm font-medium"
            >
              <FileUp className="w-4 h-4 mr-2" />
              Insertar
            </TabsTrigger>
            <TabsTrigger 
              value="layout" 
              className="h-9 px-4 data-[state=active]:bg-white data-[state=active]:text-[#2b579a] data-[state=active]:shadow-sm font-medium"
            >
              <Layout className="w-4 h-4 mr-2" />
              Diseño
            </TabsTrigger>
            <TabsTrigger 
              value="review" 
              className="h-9 px-4 data-[state=active]:bg-white data-[state=active]:text-[#2b579a] data-[state=active]:shadow-sm font-medium"
            >
              <MessageSquare className="w-4 h-4 mr-2" />
              Revisar
            </TabsTrigger>
            <TabsTrigger 
              value="view" 
              className="h-9 px-4 data-[state=active]:bg-white data-[state=active]:text-[#2b579a] data-[state=active]:shadow-sm font-medium"
            >
              <Eye className="w-4 h-4 mr-2" />
              Vista
            </TabsTrigger>
          </TabsList>

          <TabsContent value="home" className="p-3 bg-white">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 p-2 bg-[#f3f3f3] rounded-md">
                <div className="flex items-center gap-2">
                  <ToolbarButton icon={Bold} tooltip="Negrita" />
                  <ToolbarButton icon={Italic} tooltip="Cursiva" />
                  <ToolbarButton icon={Underline} tooltip="Subrayado" />
                  <Separator orientation="vertical" className="h-6" />
                  <ToolbarButton icon={AlignLeft} tooltip="Alinear a la izquierda" />
                  <ToolbarButton icon={AlignCenter} tooltip="Centrar" />
                  <ToolbarButton icon={AlignRight} tooltip="Alinear a la derecha" />
                </div>
              </div>
              <div className="flex items-center gap-2 p-2 bg-[#f3f3f3] rounded-md">
                <div className="flex items-center gap-2">
                  <ToolbarButton icon={List} tooltip="Lista con viñetas" />
                  <ToolbarButton icon={ListOrdered} tooltip="Lista numerada" />
                  <Separator orientation="vertical" className="h-6" />
                  <ToolbarButton icon={Paintbrush} tooltip="Formato de pincel" />
                  <ToolbarButton icon={Type} tooltip="Estilos" />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="insert" className="p-3 bg-white">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 p-2 bg-[#f3f3f3] rounded-md">
                <div className="flex items-center gap-2">
                  <ToolbarButton icon={Image} tooltip="Insertar imagen" />
                  <ToolbarButton icon={Table} tooltip="Insertar tabla" />
                  <ToolbarButton icon={Link} tooltip="Insertar enlace" />
                  <ToolbarButton icon={Code} tooltip="Insertar código" />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="review" className="p-3 bg-white">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 p-2 bg-[#f3f3f3] rounded-md">
                <div className="flex items-center gap-2">
                  <ToolbarButton icon={Search} tooltip="Buscar" />
                  <ToolbarButton icon={BookOpen} tooltip="Diccionario" />
                  <ToolbarButton icon={FileCheck} tooltip="Revisar ortografía" />
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Barra de IA */}
      <motion.div 
        initial={{ y: -10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="w-full bg-white border-b border-gray-200 shadow-sm"
      >
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-[#2b579a] text-white hover:bg-[#2b579a]/90 px-3 py-1">
                <Sparkles className="w-3.5 h-3.5 mr-1.5" />
                IA para Apuntes
              </Badge>
            </div>
            <div className="relative flex-1 max-w-xl">
              <Input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Escribe lo que quieres que la IA agregue al documento..."
                className="pr-10 border-[#2b579a]/30 focus:border-[#2b579a] focus:ring-[#2b579a]/20 h-10"
                onFocus={() => setShowAIPanel(true)}
              />
              <Button
                size="icon"
                className="absolute right-1 top-1/2 -translate-y-1/2 h-8 w-8 bg-[#2b579a] hover:bg-[#2b579a]/90"
                onClick={handleAIGenerate}
                disabled={loading || !prompt.trim()}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Panel de IA expandible */}
      <AnimatePresence>
        {showAIPanel && (aiText || loading) && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="w-full bg-white border-b border-gray-200 overflow-hidden shadow-sm"
          >
            <div className="max-w-7xl mx-auto px-4 py-3">
              <div className="flex items-start gap-4">
                <div className="flex-1">
                  {loading ? (
                    <div className="flex items-center gap-3 text-[#2b579a] text-sm">
                      <Progress value={33} className="w-32 h-2 bg-[#2b579a]/20" />
                      <span>Generando texto...</span>
                    </div>
                  ) : aiText ? (
                    <Card className="bg-[#f8f9fa] border-[#2b579a]/30 shadow-sm">
                      <div className="p-4">
                        <div className="text-sm text-gray-800 whitespace-pre-line leading-relaxed">{aiText}</div>
                        <div className="mt-3 flex justify-end">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={handleCopy}
                            className="text-[#2b579a] border-[#2b579a]/30 hover:bg-[#2b579a]/10 px-4"
                          >
                            <Copy className="w-3.5 h-3.5 mr-2" /> Copiar
                          </Button>
                        </div>
                      </div>
                    </Card>
                  ) : null}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-gray-500 hover:text-[#2b579a] hover:bg-[#2b579a]/10"
                  onClick={() => setShowAIPanel(false)}
                >
                  ×
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Área de documento ONLYOFFICE */}
      <div className="flex-1 w-full max-w-7xl mx-auto mt-4 px-4 pb-8">
        <Card className="overflow-hidden border-[#2b579a]/20 shadow-md bg-white">
          <div className="relative">
            {/* Regla superior */}
            <div className="absolute top-0 left-0 right-0 h-6 bg-[#f3f3f3] border-b border-gray-200 flex items-center px-4 z-20">
              <div className="flex items-center gap-8 text-xs text-gray-500">
                <span>0</span>
                <span>1</span>
                <span>2</span>
                <span>3</span>
                <span>4</span>
                <span>5</span>
                <span>6</span>
                <span>7</span>
                <span>8</span>
                <span>9</span>
                <span>10</span>
              </div>
            </div>

            {/* Regla lateral */}
            <div className="absolute top-6 left-0 w-6 bg-[#f3f3f3] border-r border-gray-200 flex flex-col items-center py-2 z-20">
              <div className="flex flex-col gap-8 text-xs text-gray-500">
                <span>0</span>
                <span>1</span>
                <span>2</span>
                <span>3</span>
                <span>4</span>
                <span>5</span>
                <span>6</span>
                <span>7</span>
                <span>8</span>
                <span>9</span>
                <span>10</span>
              </div>
            </div>

            {/* Área de edición */}
            <ScrollArea className="h-[calc(100vh-280px)]">
              <div className="relative">
                {/* Márgenes del documento */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute top-6 left-6 right-6 bottom-6 border border-gray-200 bg-white">
                    {/* Líneas de guía */}
                    <div className="absolute inset-0 bg-[linear-gradient(transparent_39px,_#e5e7eb_40px)] bg-[length:100%_40px]"></div>
                  </div>
                </div>

                {/* Editor ONLYOFFICE */}
                <div 
                  ref={editorRef} 
                  className="w-full h-full min-h-[calc(100vh-280px)] relative z-10"
                  style={{
                    marginTop: '24px',
                    marginLeft: '24px',
                    marginRight: '24px',
                    marginBottom: '24px',
                    backgroundColor: 'white',
                    boxShadow: '0 0 0 1px rgba(0,0,0,0.1)',
                  }}
                />
              </div>
            </ScrollArea>
          </div>
        </Card>
      </div>
    </div>
  );
}            