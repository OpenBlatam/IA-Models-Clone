import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Chips reutilizables
export function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center px-3 py-1 rounded-full bg-muted text-sm font-medium mr-2 mb-2 border border-border">
      {children}
    </span>
  );
}

// Sección Voz de Marca
export function BrandVoiceSection({ purpose, audience, tone, emotions }: {
  purpose: string;
  audience: string;
  tone: string[];
  emotions: string[];
}) {
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Brand Voice</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="font-semibold">Purpose <span className="text-xs font-normal text-muted-foreground">The 'why' of your communication and content.</span></div>
          <div className="mt-2 p-3 rounded-lg bg-muted/50 text-base">{purpose}</div>
        </div>
        <div>
          <div className="font-semibold">Audience <span className="text-xs font-normal text-muted-foreground">The primary people you speak to and serve.</span></div>
          <div className="mt-2 p-3 rounded-lg bg-muted/50 text-base">{audience}</div>
        </div>
        <div>
          <div className="font-semibold">Tone <span className="text-xs font-normal text-muted-foreground">The personality of how your brand sounds and feels.</span></div>
          <div className="mt-2 flex flex-wrap">
            {tone.map((t) => <Chip key={t}>{t}</Chip>)}
          </div>
        </div>
        <div>
          <div className="font-semibold">Emotions <span className="text-xs font-normal text-muted-foreground">The feelings you aim to inspire.</span></div>
          <div className="mt-2 flex flex-wrap">
            {emotions.map((e) => <Chip key={e}>{e}</Chip>)}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Sección Materiales
export function SourceMaterialsSection({ materials }: { materials: { name: string; icon?: React.ReactNode; }[] }) {
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Source Materials</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {materials.map((m, i) => (
            <li key={i} className="flex items-center gap-2">
              {m.icon}
              <span>{m.name}</span>
            </li>
          ))}
        </ul>
        <Button variant="ghost" className="mt-4 text-primary">+ Add New</Button>
      </CardContent>
    </Card>
  );
}

// Sección Imágenes y Videos
export function MediaGallerySection({ media }: { media: { type: "image" | "video"; src: string; alt?: string; }[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Images & Videos</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-4">
          {media.map((m, i) => (
            <div key={i} className="w-32 h-32 bg-muted rounded-lg flex items-center justify-center overflow-hidden">
              {m.type === "image" ? (
                <img src={m.src} alt={m.alt || "media"} className="object-cover w-full h-full" />
              ) : (
                <video src={m.src} controls className="object-cover w-full h-full" />
              )}
            </div>
          ))}
        </div>
        <Button variant="ghost" className="mt-4 text-primary">Upload</Button>
      </CardContent>
    </Card>
  );
}

// Componente principal modular
export default function BrandKitDisplay({
  brand,
  voice,
  materials,
  media,
  onCreate,
}: {
  brand: { name: string; logo: string };
  voice: { purpose: string; audience: string; tone: string[]; emotions: string[] };
  materials: { name: string; icon?: React.ReactNode }[];
  media: { type: "image" | "video"; src: string; alt?: string }[];
  onCreate?: () => void;
}) {
  return (
    <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8">
      <div>
        <div className="flex items-center gap-4 mb-6">
          <img src={brand.logo} alt={brand.name} className="w-14 h-14 rounded-full shadow" />
          <span className="text-2xl font-bold">{brand.name}</span>
        </div>
        <BrandVoiceSection {...voice} />
      </div>
      <div>
        <div className="flex justify-end mb-6">
          <Button onClick={onCreate} className="bg-black text-white px-6 py-2 rounded-lg shadow hover:bg-gray-900">
            Create with this Brand Kit
          </Button>
        </div>
        <SourceMaterialsSection materials={materials} />
        <MediaGallerySection media={media} />
      </div>
    </div>
  );
} 