'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Figma, FileText, Users, Plus, Settings, Share2, Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { useRouter } from 'next/navigation';

interface Design {
  id: string;
  title: string;
  lastModified: string;
  thumbnail: string;
  category: string;
  likes: number;
}

export default function ColaboracionIAPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('design');

  // Mock data for designs with real Unsplash images
  const designs: Design[] = [
    { 
      id: '1', 
      title: 'Diseño Minimalista', 
      lastModified: '2024-03-20',
      thumbnail: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1000',
      category: 'Minimalista',
      likes: 245
    },
    { 
      id: '2', 
      title: 'Fondo Corporativo', 
      lastModified: '2024-03-19',
      thumbnail: 'https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=1000',
      category: 'Corporativo',
      likes: 189
    },
    { 
      id: '3', 
      title: 'Diseño Creativo', 
      lastModified: '2024-03-18',
      thumbnail: 'https://images.unsplash.com/photo-1618005198919-d3d4b5a92ead?q=80&w=1000',
      category: 'Creativo',
      likes: 312
    },
    { 
      id: '4', 
      title: 'Fondo Tecnológico', 
      lastModified: '2024-03-17',
      thumbnail: 'https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=1000',
      category: 'Tecnología',
      likes: 278
    },
    { 
      id: '5', 
      title: 'Diseño Abstracto', 
      lastModified: '2024-03-16',
      thumbnail: 'https://images.unsplash.com/photo-1618005198919-d3d4b5a92ead?q=80&w=1000',
      category: 'Abstracto',
      likes: 156
    },
    { 
      id: '6', 
      title: 'Fondo Moderno', 
      lastModified: '2024-03-15',
      thumbnail: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1000',
      category: 'Moderno',
      likes: 423
    },
  ];

  const handleDesignClick = (id: string) => {
    router.push(`/dashboard/colaboracion-ia/${id}`);
  };

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold tracking-tight">Colaboración IA</h2>
          <p className="text-sm text-muted-foreground">
            Diseña y crea presentaciones con la ayuda de IA
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Share2 className="h-4 w-4" />
            Compartir
          </Button>
          <Button size="sm" className="gap-2">
            <Plus className="h-4 w-4" />
            Nuevo Proyecto
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="design" className="flex-1 space-y-4">
        <TabsList>
          <TabsTrigger value="design" className="gap-2">
            <Figma className="h-4 w-4" />
            Diseño
          </TabsTrigger>
          <TabsTrigger value="slides" className="gap-2">
            <FileText className="h-4 w-4" />
            Slides
          </TabsTrigger>
        </TabsList>

        <TabsContent value="design" className="space-y-4">
          {/* Search and Filter */}
          <div className="flex items-center gap-4">
            <Input
              placeholder="Buscar diseños..."
              className="max-w-sm"
            />
            <Button variant="outline" size="sm">
              Filtros
            </Button>
          </div>

          {/* Design Grid */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {designs.map((design) => (
              <Card 
                key={design.id} 
                className="group cursor-pointer overflow-hidden transition-all hover:shadow-lg"
                onClick={() => handleDesignClick(design.id)}
              >
                <div className="relative aspect-video overflow-hidden">
                  <img
                    src={design.thumbnail}
                    alt={design.title}
                    className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
                  <div className="absolute bottom-2 left-2 flex items-center gap-2">
                    <Badge variant="secondary" className="bg-white/90 text-black">
                      {design.category}
                    </Badge>
                    <div className="flex items-center gap-1 text-white">
                      <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                      <span className="text-sm">{design.likes}</span>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  <h3 className="font-medium">{design.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    Última modificación: {design.lastModified}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="slides" className="space-y-4">
          {/* Search and Filter */}
          <div className="flex items-center gap-4">
            <Input
              placeholder="Buscar presentaciones..."
              className="max-w-sm"
            />
            <Button variant="outline" size="sm">
              Filtros
            </Button>
          </div>

          {/* Slides Grid */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {designs.map((design) => (
              <Card 
                key={design.id} 
                className="group cursor-pointer overflow-hidden transition-all hover:shadow-lg"
                onClick={() => handleDesignClick(design.id)}
              >
                <div className="relative aspect-video overflow-hidden">
                  <img
                    src={design.thumbnail}
                    alt={design.title}
                    className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
                  <div className="absolute bottom-2 left-2 flex items-center gap-2">
                    <Badge variant="secondary" className="bg-white/90 text-black">
                      {design.category}
                    </Badge>
                    <div className="flex items-center gap-1 text-white">
                      <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                      <span className="text-sm">{design.likes}</span>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  <h3 className="font-medium">{design.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    Última modificación: {design.lastModified}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
} 