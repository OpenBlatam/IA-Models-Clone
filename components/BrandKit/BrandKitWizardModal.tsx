import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const templates = [
  {
    id: 1,
    title: "Transformación y Oportunidades con IA",
    subtitle: "Explora el futuro de la IA en educación",
    color: "#FF6FA5",
    image: "/template1.jpg",
    cta: "Únete hoy"
  },
  {
    id: 2,
    title: "Únete a la revolución educativa",
    subtitle: "",
    color: "#FFD6E0",
    image: "/template2.jpg",
    cta: "Prepárate ya"
  },
  {
    id: 3,
    title: "Prepárate para el futuro con AI.",
    subtitle: "",
    color: "#B2F0F0",
    image: "/template3.jpg",
    cta: "Únete a Bitam hoy."
  },
  {
    id: 4,
    title: "¡Participa! Descubre las ventajas.",
    subtitle: "",
    color: "#F39AC1",
    image: "/template4.jpg",
    cta: "Únete ahora mismo!"
  }
];

export function BrandKitWizardModal({ open, onClose, onGenerate, brandKit }) {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Refinar detalles del post</DialogTitle>
          <div className="text-muted-foreground mb-2">Las sugerencias se basan en el tema de tu publicación</div>
        </DialogHeader>
        <div className="mb-4">
          <div className="font-medium mb-2 flex items-center justify-between">
            <span>Selecciona un diseño para comenzar <span className="text-red-500">*</span></span>
            <button className="text-primary text-sm font-medium hover:underline">Ver todos los templates</button>
          </div>
          <div className="flex flex-col md:flex-row gap-6 items-center justify-center">
            {/* Template seleccionado grande */}
            {selected !== null && (
              <div className="flex flex-col items-center">
                <div className="rounded-2xl border-2 border-primary shadow-lg overflow-hidden mb-2 w-[260px] h-[260px] md:w-[320px] md:h-[320px] bg-white flex items-center justify-center">
                  <img
                    src={templates[selected].image}
                    alt={templates[selected].title}
                    className="object-cover w-full h-full"
                  />
                </div>
                <div className="font-bold text-lg mb-1 text-center max-w-xs truncate">{templates[selected].title}</div>
                <div className="text-sm mb-2 text-center max-w-xs truncate">{templates[selected].subtitle}</div>
                <div className="text-xs font-medium mt-auto text-center">{templates[selected].cta}</div>
              </div>
            )}
            {/* Thumbnails de los demás templates */}
            <div className="flex flex-col items-center justify-center">
              <div className="grid grid-cols-3 grid-rows-1 gap-3 justify-center items-center">
                {templates.slice(0, 3).map((tpl, idx) => (
                  <div
                    key={tpl.id}
                    className={`rounded-2xl border-2 cursor-pointer min-w-[70px] max-w-[80px] min-h-[70px] max-h-[80px] p-1 flex flex-col items-center transition-all ${
                      selected === idx ? "border-primary shadow-lg" : "border-transparent hover:border-primary hover:shadow-lg"
                    } bg-white`}
                    style={{ background: tpl.color }}
                    onClick={() => setSelected(idx)}
                  >
                    <img src={tpl.image} alt={tpl.title} className="rounded-xl w-full h-full object-cover mb-1" />
                    {selected === idx && (
                      <div className="font-bold text-xs mb-0.5 truncate w-full text-center">{tpl.title}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>Cancelar</Button>
          <Button
            disabled={selected === null}
            onClick={() => onGenerate(templates[selected!])}
          >
            Generar Post
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
} 