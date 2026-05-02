import React from "react";
import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Pagination } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/pagination';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "@/components/ui/dialog";

const templates = [
  {
    id: 5,
    title: "AI Code Editor",
    color: "#fff",
    image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-b94a933979ae43ed9be5.png",
  },
  {
    id: 6,
    title: "AI Code Reel",
    color: "#fff",
    image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-9ffaebf7b3a9532cbb8a.png",
    reel: true,
  },
  {
    id: 7,
    title: "AI Code Slide",
    color: "#fff",
    image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/s2znj8fzokoxjm9lfmsb2wklocnc5yzh_k3-p9x855kodfzg3-6eiu-09e3db4cf0b1e62c6ccc.png",
    reel: true,
  },
  {
    id: 8,
    title: "Minimalist Fashion",
    color: "#fff",
    image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/s8ygb5ebwfhbkuenmxr0mzigzwe56e5d_wwlv8h8dhhdzee5fnkpx8_thumbnail_1748523407-76bfa6e00fac762bc1e3.png",
  },
  {
    id: 9,
    title: "Corporate Collaboration",
    color: "#fff",
    image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/fayp2jtpwoxghtkqyg2vrmddhchmpitp_5scussplt00-pse6wuzan_thumbnail_1744740805-b368b259aac04428e3e2.png",
  },
  {
    id: 1,
    title: "Educativo Moderno",
    color: "#FF6FA5",
    image: "/template1.jpg",
  },
  {
    id: 2,
    title: "Tecnología Futurista",
    color: "#FFD6E0",
    image: "/template2.jpg",
  },
  {
    id: 3,
    title: "Negocios Minimal",
    color: "#B2F0F0",
    image: "/template3.jpg",
  },
  {
    id: 4,
    title: "Creativo Vibrante",
    color: "#F39AC1",
    image: "/template4.jpg",
  },
];

// Agrupación de templates por categoría
export const templateCategories = [
  {
    title: 'Elegant',
    templates: [
      {
        id: 5,
        title: "Manufacturing Excellence Template",
        subtitle: "Instagram Square Post by Blaze.ai",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-b94a933979ae43ed9be5.png",
      },
      {
        id: 6,
        title: "Running Motivation Template",
        subtitle: "Instagram Square Post by Blaze.ai",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-9ffaebf7b3a9532cbb8a.png",
      },
      {
        id: 7,
        title: "Elegant Love Quote Template",
        subtitle: "Instagram Square Post by Blaze.ai",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/s2znj8fzokoxjm9lfmsb2wklocnc5yzh_k3-p9x855kodfzg3-6eiu-09e3db4cf0b1e62c6ccc.png",
      },
      {
        id: 8,
        title: "Elegant Fashion Promotion Template",
        subtitle: "Instagram Square Post by Blaze.ai",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/s8ygb5ebwfhbkuenmxr0mzigzwe56e5d_wwlv8h8dhhdzee5fnkpx8_thumbnail_1748523407-76bfa6e00fac762bc1e3.png",
      },
    ],
  },
  {
    title: 'Feminine',
    templates: [
      {
        id: 10,
        title: "Inspirational Quote Template",
        subtitle: "LinkedIn Visual Post by Blaze.ai",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/fayp2jtpwoxghtkqyg2vrmddhchmpitp_5scussplt00-pse6wuzan_thumbnail_1744740805-b368b259aac04428e3e2.png",
      },
      {
        id: 11,
        title: "Minimalist Promotional Quote",
        subtitle: "LinkedIn Visual Post by Blaze.ai",
        image: "/template1.jpg",
      },
      {
        id: 12,
        title: "Fall Pumpkin Collage Template",
        subtitle: "LinkedIn Visual Post by Blaze.ai",
        image: "/template2.jpg",
      },
      {
        id: 308,
        title: "Template Visible 8",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/phwsjcbwiu9mdhsl5rnlh2gxxggbctsu_ekuiuwwjp6waehsvf5dnr-c471c05ac96352b45713.png",
      },
      {
        id: 309,
        title: "Template Visible 9",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/rup4zgnkivysuobvgms3dbanzzbsebae_lnmhzl7juw-ggpfrw7t9y-967faf47396f8e42f60d.png",
      },
      {
        id: 310,
        title: "Template Visible 10",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/0jbgqolilsfzcuwxr67crmnr9dhz3mkp_dn0hh2dlvqwlvt2aofcjk-02a057f364f9c317b680.png",
      },
    ],
  },
  {
    title: 'Nuevos Templates',
    templates: [
      {
        id: 301,
        title: "Template Visible 1",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-b94a933979ae43ed9be5.png",
      },
      {
        id: 302,
        title: "Template Visible 2",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-9ffaebf7b3a9532cbb8a.png",
      },
      {
        id: 303,
        title: "Template Visible 3",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/s2znj8fzokoxjm9lfmsb2wklocnc5yzh_k3-p9x855kodfzg3-6eiu-09e3db4cf0b1e62c6ccc.png",
      },
      {
        id: 304,
        title: "Template Visible 4",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/s8ygb5ebwfhbkuenmxr0mzigzwe56e5d_wwlv8h8dhhdzee5fnkpx8_thumbnail_1748523407-76bfa6e00fac762bc1e3.png",
      },
      {
        id: 305,
        title: "Template Visible 5",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/fayp2jtpwoxghtkqyg2vrmddhchmpitp_5scussplt00-pse6wuzan_thumbnail_1744740805-b368b259aac04428e3e2.png",
      },
      {
        id: 306,
        title: "Template Visible 6",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/article_uploads//template-bf8f55bb359f8f707f0b.png",
      },
      {
        id: 307,
        title: "Template Visible 7",
        subtitle: "Ejemplo Cloudinary",
        image: "https://res.cloudinary.com/almanac/image/upload/q_auto,c_scale,w_952/uploads/ker3leneqbjnrcr0y0iiltg5uouqcnxe_f5sb1ye_rvr5pfpl493e1-aef94a016831808f22a6.png",
      },
    ],
  },
];

export function TemplateCard({ template }: { template: any }) {
  return (
    <div className="flex flex-col items-center w-full">
      <div className="rounded-2xl overflow-hidden bg-white shadow-md w-full aspect-square flex items-center justify-center">
        <img
          src={template.image}
          alt={template.title}
          className="object-cover w-full h-full"
        />
      </div>
      <div className="mt-3 w-full text-left">
        <div className="font-semibold text-base leading-tight truncate" title={template.title}>{template.title}</div>
        <div className="text-xs text-muted-foreground truncate" title={template.subtitle}>{template.subtitle}</div>
      </div>
    </div>
  );
}

export function TemplateCarousel({ title, templates, onCardClick }: { title: string; templates: any[]; onCardClick?: (tpl: any) => void }) {
  return (
    <section className="mb-10">
      <h2 className="text-2xl font-bold mb-4">{title}</h2>
      <Swiper
        modules={[Navigation, Pagination]}
        spaceBetween={24}
        slidesPerView={3}
        navigation
        pagination={{ clickable: true }}
        breakpoints={{
          640: { slidesPerView: 1.2 },
          768: { slidesPerView: 2 },
          1024: { slidesPerView: 3 },
          1280: { slidesPerView: 4 },
        }}
        className="w-full"
      >
        {templates.map((tpl) => (
          <SwiperSlide key={tpl.id}>
            <div onClick={() => onCardClick?.(tpl)} className="cursor-pointer">
              <TemplateCard template={tpl} />
            </div>
          </SwiperSlide>
        ))}
      </Swiper>
    </section>
  );
}

export function TemplateDetailModal({ open, onClose, template, onUseTemplate }: {
  open: boolean;
  onClose: () => void;
  template: any;
  onUseTemplate?: (tpl: any) => void;
}) {
  if (!template) return null;
  // Datos de ejemplo para demo visual
  const details = {
    format: template.format || "Instagram Square Post",
    category: template.category || "Food",
    type: template.type || ["Photo", "Collage", "Social Media"],
    style: template.style || ["Modern", "Minimalist", "Simple", "Aesthetic", "Clean"],
    replaceable: template.replaceable !== undefined ? template.replaceable : false,
    colors: template.colors || ["#fff", "#bcd2e8", "#7b8fa1", "#f8f8f8", "#000"],
  };
  return (
    <Dialog open={open} onOpenChange={v => !v && onClose()}>
      <DialogContent className="max-w-3xl w-full p-0 overflow-hidden">
        <div className="flex flex-col md:flex-row gap-0 md:gap-8 w-full">
          {/* Imagen grande */}
          <div className="flex-1 flex items-center justify-center bg-muted p-6">
            <img
              src={template.image}
              alt={template.title}
              className="rounded-2xl w-full max-w-md aspect-square object-cover shadow-lg"
            />
          </div>
          {/* Detalles */}
          <div className="flex-1 min-w-[320px] p-6 flex flex-col gap-4">
            <div className="flex items-center justify-between gap-2">
              <DialogTitle className="text-2xl font-bold flex-1">
                <span className="inline-flex items-center gap-2">
                  <span className="text-pink-500">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="6" fill="#fff"/><path d="M7.5 12C7.5 9.51472 9.51472 7.5 12 7.5C14.4853 7.5 16.5 9.51472 16.5 12C16.5 14.4853 14.4853 16.5 12 16.5C9.51472 16.5 7.5 14.4853 7.5 12Z" fill="#F56040"/><path d="M12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2ZM12 20C7.58172 20 4 16.4183 4 12C4 7.58172 7.58172 4 12 4C16.4183 4 20 7.58172 20 12C20 16.4183 16.4183 20 12 20Z" fill="#F56040"/></svg>
                  </span>
                  {template.title}
                </span>
              </DialogTitle>
              <div className="flex gap-2">
                <button className="px-3 py-1 rounded-md bg-muted text-xs font-medium border hover:bg-accent" onClick={() => navigator.clipboard.writeText(template.image)}>Copy Link</button>
                <button className="px-3 py-1 rounded-md bg-primary text-xs font-medium text-white hover:bg-primary/90" onClick={() => { onUseTemplate?.(template); onClose(); }}>Use This Template</button>
              </div>
            </div>
            <div className="flex flex-col gap-2 mt-2">
              <div className="text-sm font-medium text-muted-foreground">Format</div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-muted text-sm">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="6" fill="#fff"/><path d="M7.5 12C7.5 9.51472 9.51472 7.5 12 7.5C14.4853 7.5 16.5 9.51472 16.5 12C16.5 14.4853 14.4853 16.5 12 16.5C9.51472 16.5 7.5 14.4853 7.5 12Z" fill="#F56040"/></svg>
                  {details.format}
                </span>
              </div>
              <div className="text-sm font-medium text-muted-foreground mt-2">Category</div>
              <div className="flex flex-wrap gap-2">
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-muted text-sm">{details.category}</span>
              </div>
              <div className="text-sm font-medium text-muted-foreground mt-2">Type</div>
              <div className="flex flex-wrap gap-2">
                {details.type.map((t: string) => (
                  <span key={t} className="inline-flex items-center gap-1 px-2 py-1 rounded bg-muted text-sm">{t}</span>
                ))}
              </div>
              <div className="text-sm font-medium text-muted-foreground mt-2">Style</div>
              <div className="flex flex-wrap gap-2">
                {details.style.map((s: string) => (
                  <span key={s} className="inline-flex items-center gap-1 px-2 py-1 rounded bg-muted text-sm">{s}</span>
                ))}
              </div>
              <div className="text-sm font-medium text-muted-foreground mt-2">Replaceable image</div>
              <div className="flex items-center gap-2">
                {details.replaceable ? (
                  <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-white">✓</span>
                ) : (
                  <span className="w-6 h-6 rounded-full bg-muted flex items-center justify-center text-xl">✗</span>
                )}
              </div>
              <div className="text-sm font-medium text-muted-foreground mt-2">Colors</div>
              <div className="flex items-center gap-2">
                {details.colors.map((c: string, i: number) => (
                  <span key={i} className="w-6 h-6 rounded-full border" style={{ background: c }} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
} 