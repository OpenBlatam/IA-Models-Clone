import React, { useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";

const templates = [
  { id: 1, title: "Mejora tus habilidades con cursos de blockchain.", image: "/template1.jpg" },
  { id: 2, title: "Transforma tu futuro", image: "/template2.jpg" },
  { id: 3, title: "Únete hoy!", image: "/template3.jpg" },
  { id: 4, title: "Aprende tecnología mejor!", image: "/template4.jpg" },
  { id: 5, title: "Únete a la revolución blockchain.", image: "/template5.jpg" },
  { id: 6, title: "Innovación blockchain", image: "/template6.jpg" },
  { id: 7, title: "¡Listo para el siguiente nivel?", image: "/template7.jpg" },
  { id: 8, title: "Mejora tus habilidades en tecnología!", image: "/template8.jpg" },
];

export function RefinePostDetailsModal({
  open,
  onClose,
  onBack,
  onGenerate,
  initialCaption = "",
  loading,
}: {
  open: boolean;
  onClose: () => void;
  onBack: () => void;
  onGenerate: (templateId: number, caption: string) => void;
  initialCaption?: string;
  loading?: boolean;
}) {
  const [selected, setSelected] = useState<number>(templates[0].id);
  const [caption, setCaption] = useState(initialCaption);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl p-10 rounded-2xl bg-white shadow-xl space-y-8">
        <DialogTitle className="text-3xl font-bold text-gray-900 mb-2">Refine post details</DialogTitle>
        <div className="text-gray-500 text-base mb-4">Suggestions are based on the topic of your post</div>
        <div className="mb-2 font-medium text-lg">Select a design you'd like to start with <span className="text-red-500">*</span> <span className="float-right text-primary cursor-pointer font-medium text-base hover:underline">See All Templates</span></div>
        <div className="grid grid-cols-4 gap-4 mb-6">
          {templates.map(tpl => (
            <div
              key={tpl.id}
              className={`rounded-2xl border-2 cursor-pointer p-2 flex flex-col items-center transition-all ${selected === tpl.id ? "border-[#0099FF] shadow-lg" : "border-transparent"}`}
              onClick={() => setSelected(tpl.id)}
              style={{ background: selected === tpl.id ? "#F6F7F9" : "#fff" }}
            >
              <img src={tpl.image} alt={tpl.title} className="rounded-xl w-full h-28 object-cover mb-2" />
              <div className="font-bold text-base text-center mb-1">{tpl.title}</div>
            </div>
          ))}
        </div>
        <div className="mb-2 font-medium text-base">Edit the caption beneath your post <span className="text-red-500">*</span></div>
        <div className="flex items-center gap-2 mb-2">
          <textarea
            value={caption}
            onChange={e => setCaption(e.target.value)}
            className="w-full text-lg leading-normal px-4 py-3 border-2 border-[#0099FF] focus:border-[#0099FF] rounded-lg bg-white shadow-sm resize-none"
            placeholder="Escribe el caption para tu post..."
            rows={2}
            style={{ fontFamily: "inherit" }}
          />
          <Button
            variant="link"
            onClick={() => setCaption("Generando caption IA...")}
            disabled={loading}
            className="pl-2 text-[#0099FF] font-medium flex items-center"
            style={{ minWidth: 0 }}
          >
            <RefreshCw className="w-5 h-5 mr-1" />
            Regenerate
          </Button>
        </div>
        <div className="flex justify-between items-center mt-4">
          <Button variant="ghost" onClick={onBack} className="text-gray-500">Purpose & Goals</Button>
          <Button
            className="h-12 px-8 text-lg rounded-lg bg-black text-white hover:bg-gray-900 font-semibold shadow"
            disabled={!caption || loading}
            onClick={() => onGenerate(selected, caption)}
          >
            Generate Post
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
} 