'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Figma, FileText, Users, Plus, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';

interface Slide {
  id: string;
  title: string;
  lastModified: string;
}

export default function CollaborationSidebar() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'design' | 'slides'>('design');

  // Mock data for slides
  const slides: Slide[] = [
    { id: '1', title: 'Presentación Principal', lastModified: '2024-03-20' },
    { id: '2', title: 'Diseño de UI/UX', lastModified: '2024-03-19' },
    { id: '3', title: 'Wireframes', lastModified: '2024-03-18' },
  ];

  return (
    <motion.div
      initial={{ width: 48 }}
      animate={{ width: isExpanded ? 280 : 48 }}
      className="relative h-full bg-background border-r"
    >
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex h-14 items-center justify-between px-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsExpanded(!isExpanded)}
            className="h-8 w-8"
          >
            <Figma className="h-4 w-4" />
          </Button>
          {isExpanded && (
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {/* Tabs */}
        {isExpanded && (
          <div className="flex border-b px-2">
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                "flex-1 justify-start gap-2",
                activeTab === 'design' && "bg-accent"
              )}
              onClick={() => setActiveTab('design')}
            >
              <Figma className="h-4 w-4" />
              Diseño
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                "flex-1 justify-start gap-2",
                activeTab === 'slides' && "bg-accent"
              )}
              onClick={() => setActiveTab('slides')}
            >
              <FileText className="h-4 w-4" />
              Slides
            </Button>
          </div>
        )}

        {/* Content */}
        <ScrollArea className="flex-1">
          {isExpanded && (
            <div className="p-2">
              {activeTab === 'design' ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">Diseños Recientes</h3>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="space-y-1">
                    {slides.map((slide) => (
                      <Button
                        key={slide.id}
                        variant="ghost"
                        className="w-full justify-start gap-2"
                      >
                        <Figma className="h-4 w-4" />
                        <div className="flex flex-col items-start">
                          <span className="text-sm">{slide.title}</span>
                          <span className="text-xs text-muted-foreground">
                            {slide.lastModified}
                          </span>
                        </div>
                      </Button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">Presentaciones</h3>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="space-y-1">
                    {slides.map((slide) => (
                      <Button
                        key={slide.id}
                        variant="ghost"
                        className="w-full justify-start gap-2"
                      >
                        <FileText className="h-4 w-4" />
                        <div className="flex flex-col items-start">
                          <span className="text-sm">{slide.title}</span>
                          <span className="text-xs text-muted-foreground">
                            {slide.lastModified}
                          </span>
                        </div>
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </ScrollArea>

        {/* Footer */}
        {isExpanded && (
          <div className="border-t p-2">
            <Button variant="ghost" className="w-full justify-start gap-2">
              <Users className="h-4 w-4" />
              Colaboradores
            </Button>
          </div>
        )}
      </div>
    </motion.div>
  );
}    