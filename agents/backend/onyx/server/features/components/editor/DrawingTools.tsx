'use client';

import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { EDITOR_CONSTANTS } from '@/lib/editor/constants';
import {
  Pencil,
  Square,
  Circle,
  Type,
  Image as ImageIcon,
  ArrowRight,
  Minus,
  Star,
  Hexagon,
  Eraser,
  Droplet,
  Palette,
  Layers,
} from 'lucide-react';

interface DrawingToolsProps {
  onSelectTool: (tool: string) => void;
  onColorChange: (color: string) => void;
  onOpacityChange: (opacity: number) => void;
  onLineStyleChange: (style: string) => void;
  onShadowChange: (shadow: string) => void;
  selectedTool: string;
  currentColor: string;
  currentOpacity: number;
  currentLineStyle: string;
  currentShadow: string;
}

export function DrawingTools({
  onSelectTool,
  onColorChange,
  onOpacityChange,
  onLineStyleChange,
  onShadowChange,
  selectedTool,
  currentColor,
  currentOpacity,
  currentLineStyle,
  currentShadow,
}: DrawingToolsProps) {
  const tools = [
    { id: 'select', icon: Pencil, label: 'Seleccionar' },
    { id: 'rectangle', icon: Square, label: 'Rectángulo' },
    { id: 'ellipse', icon: Circle, label: 'Círculo' },
    { id: 'text', icon: Type, label: 'Texto' },
    { id: 'image', icon: ImageIcon, label: 'Imagen' },
    { id: 'arrow', icon: ArrowRight, label: 'Flecha' },
    { id: 'line', icon: Minus, label: 'Línea' },
    { id: 'star', icon: Star, label: 'Estrella' },
    { id: 'polygon', icon: Hexagon, label: 'Polígono' },
    { id: 'eraser', icon: Eraser, label: 'Borrador' },
  ];

  return (
    <div className="flex flex-col gap-2 p-2 bg-background border rounded-lg shadow-sm">
      <div className="flex flex-wrap gap-1">
        {tools.map((tool) => (
          <TooltipProvider key={tool.id}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={selectedTool === tool.id ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => onSelectTool(tool.id)}
                >
                  <tool.icon className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{tool.label}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </div>

      <Separator />

      <div className="flex items-center gap-2">
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon">
              <Droplet className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-64">
            <div className="grid gap-4">
              <div className="space-y-2">
                <Label>Color</Label>
                <div className="grid grid-cols-6 gap-2">
                  {Object.values(EDITOR_CONSTANTS.DEFAULT_COLORS).map((color) => (
                    <Button
                      key={color}
                      variant="ghost"
                      size="icon"
                      className="w-8 h-8"
                      style={{ backgroundColor: color }}
                      onClick={() => onColorChange(color)}
                    />
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <Label>Opacidad</Label>
                <Slider
                  value={[currentOpacity]}
                  onValueChange={([value]) => onOpacityChange(value)}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>

        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon">
              <Palette className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-64">
            <div className="grid gap-4">
              <div className="space-y-2">
                <Label>Estilo de Línea</Label>
                <div className="flex gap-2">
                  {Object.values(EDITOR_CONSTANTS.DEFAULT_STYLES.LINE_STYLES).map((style) => (
                    <Button
                      key={style}
                      variant={currentLineStyle === style ? 'default' : 'ghost'}
                      onClick={() => onLineStyleChange(style)}
                    >
                      {style}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <Label>Sombra</Label>
                <div className="flex gap-2">
                  {Object.values(EDITOR_CONSTANTS.DEFAULT_STYLES.SHADOW_TYPES).map((shadow) => (
                    <Button
                      key={shadow}
                      variant={currentShadow === shadow ? 'default' : 'ghost'}
                      onClick={() => onShadowChange(shadow)}
                    >
                      {shadow}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          </PopoverContent>
        </Popover>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon">
                <Layers className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Capas</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}     