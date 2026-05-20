"use client"

import { useState, useRef, useEffect } from "react"
import { DynamicKonva as Stage, DynamicLayer as Layer, DynamicImage as Image, DynamicGroup as Group } from "@/components/konva-wrapper"
import useImage from "use-image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Loader2, Wand2 } from "lucide-react"

interface ProductBuilderProps {
  initialCategory?: string
}

const GAME_CONFIGS = {
  bebida: {
    title: "Constructor de Bebidas",
    colors: ["#FF5733", "#33FF57", "#3357FF", "#F3FF33"],
    sizes: ["small", "medium", "large"],
    extras: ["Hielo", "Crema", "Chocolate", "Caramelo"],
    baseImage: "/images/products/base-cup.svg",
    lidImage: "/images/products/lid.svg",
    strawImage: "/images/products/straw.svg"
  },
  oficina: {
    title: "Constructor de Oficina",
    colors: ["#8B4513", "#2F4F4F", "#4B0082", "#800000"],
    sizes: ["compacto", "estándar", "espacioso"],
    extras: ["Plantas", "Lámparas", "Decoración", "Almacenamiento"],
    baseImage: "/images/products/desk.svg",
    lidImage: "/images/products/chair.svg",
    strawImage: "/images/products/accessories.svg"
  },
  ropa: {
    title: "Constructor de Ropa",
    colors: ["#000000", "#FFFFFF", "#FF0000", "#0000FF"],
    sizes: ["XS", "S", "M", "L", "XL"],
    extras: ["Accesorios", "Estampados", "Bolsillos", "Detalles"],
    baseImage: "/images/products/shirt.svg",
    lidImage: "/images/products/pants.svg",
    strawImage: "/images/products/accessories.svg"
  },
  avatar: {
    title: "Constructor de Avatar",
    colors: ["#FFD700", "#C0C0C0", "#CD7F32", "#000000"],
    sizes: ["pequeño", "mediano", "grande"],
    extras: ["Sombreros", "Gafas", "Accesorios", "Expresiones"],
    baseImage: "/images/products/face.svg",
    lidImage: "/images/products/hair.svg",
    strawImage: "/images/products/accessories.svg"
  }
}

export function ProductBuilder({ initialCategory = "bebida" }: ProductBuilderProps) {
  const [selectedCategory, setSelectedCategory] = useState(initialCategory)
  const [customizations, setCustomizations] = useState({
    color: GAME_CONFIGS[initialCategory as keyof typeof GAME_CONFIGS].colors[0],
    size: GAME_CONFIGS[initialCategory as keyof typeof GAME_CONFIGS].sizes[0],
    extras: [] as string[],
  })
  
  const [prompt, setPrompt] = useState("")
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  
  const stageRef = useRef(null)
  const config = GAME_CONFIGS[selectedCategory as keyof typeof GAME_CONFIGS]
  const [baseImage] = useImage(config.baseImage)
  const [lidImage] = useImage(config.lidImage)
  const [strawImage] = useImage(config.strawImage)
  const [aiImage] = useImage(generatedImage || "")

  useEffect(() => {
    setSelectedCategory(initialCategory)
    setCustomizations({
      color: GAME_CONFIGS[initialCategory as keyof typeof GAME_CONFIGS].colors[0],
      size: GAME_CONFIGS[initialCategory as keyof typeof GAME_CONFIGS].sizes[0],
      extras: []
    })
  }, [initialCategory])

  const handleDragEnd = (e: any) => {
    const node = e.target
    const newCustomizations = { ...customizations }
    setCustomizations(newCustomizations)
  }

  const handleColorChange = (color: string) => {
    setCustomizations(prev => ({ ...prev, color }))
  }

  const handleSizeChange = (size: string) => {
    setCustomizations(prev => ({ ...prev, size }))
  }

  const handleExtraAdd = (extra: string) => {
    setCustomizations(prev => ({
      ...prev,
      extras: prev.extras.includes(extra)
        ? prev.extras.filter(e => e !== extra)
        : [...prev.extras, extra]
    }))
  }

  const generateImage = async () => {
    if (!prompt) return

    setIsGenerating(true)
    try {
      const response = await fetch("https://openrouter.ai/api/v1/images/generations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.NEXT_PUBLIC_OPENROUTER_API_KEY}`,
          "HTTP-Referer": window.location.origin,
        },
        body: JSON.stringify({
          model: "stability-ai/sdxl",
          prompt: `${prompt}, ${selectedCategory}, ${customizations.color}, ${customizations.size}, ${customizations.extras.join(", ")}`,
          n: 1,
          size: "1024x1024",
        }),
      })

      const data = await response.json()
      if (data.data?.[0]?.url) {
        setGeneratedImage(data.data[0].url)
      }
    } catch (error) {
      console.error("Error generating image:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-center">
        <div className="w-96 h-96 border border-gray-200 rounded-lg bg-white flex items-center justify-center">
          <p className="text-gray-500">Product Builder (Canvas disabled)</p>
        </div>
      </div>

      <Card className="p-4">
        <div className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Describe tu producto ideal..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <Button
              onClick={generateImage}
              disabled={isGenerating || !prompt}
            >
              {isGenerating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wand2 className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </Card>

      <div className="space-y-4">
        <div>
          <h4 className="font-medium mb-2">Color</h4>
          <div className="flex gap-2">
            {config.colors.map((color) => (
              <button
                key={color}
                className="w-8 h-8 rounded-full border-2 border-gray-200 hover:scale-110 transition-transform"
                style={{ backgroundColor: color }}
                onClick={() => handleColorChange(color)}
              />
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-medium mb-2">Tamaño</h4>
          <div className="flex gap-2">
            {config.sizes.map((size) => (
              <button
                key={size}
                className={`px-4 py-2 rounded-lg border transition-colors ${
                  customizations.size === size
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-200 hover:border-blue-200"
                }`}
                onClick={() => handleSizeChange(size)}
              >
                {size.charAt(0).toUpperCase() + size.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-medium mb-2">Extras</h4>
          <div className="flex flex-wrap gap-2">
            {config.extras.map((extra) => (
              <button
                key={extra}
                className={`px-4 py-2 rounded-lg border transition-colors ${
                  customizations.extras.includes(extra)
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-200 hover:border-blue-200"
                }`}
                onClick={() => handleExtraAdd(extra)}
              >
                {extra}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}   