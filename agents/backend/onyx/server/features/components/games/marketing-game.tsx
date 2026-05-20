"use client"

import { useState, useEffect } from "react"
import { Stage, Layer, Text, Group, Circle, Rect } from "react-konva"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface CampaignMetrics {
  reach: number
  engagement: number
  conversion: number
  roi: number
}

interface MarketingGameProps {
  initialBudget?: number
}

export function MarketingGame({ initialBudget = 1000 }: MarketingGameProps) {
  const [budget, setBudget] = useState(initialBudget)
  const [selectedChannel, setSelectedChannel] = useState<string>("")
  const [campaignName, setCampaignName] = useState("")
  const [targetAudience, setTargetAudience] = useState("")
  const [metrics, setMetrics] = useState<CampaignMetrics>({
    reach: 0,
    engagement: 0,
    conversion: 0,
    roi: 0
  })
  const [feedback, setFeedback] = useState<string[]>([])
  const [isSimulating, setIsSimulating] = useState(false)
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const updateSize = () => {
      const container = document.getElementById('stage-container')
      if (container) {
        setStageSize({
          width: container.offsetWidth,
          height: Math.min(container.offsetWidth * 0.5, 400)
        })
      }
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  const channels = [
    { id: "social", name: "Redes Sociales", cost: 300, reach: 0.7, engagement: 0.8 },
    { id: "email", name: "Email Marketing", cost: 200, reach: 0.5, engagement: 0.6 },
    { id: "search", name: "Búsqueda", cost: 400, reach: 0.6, engagement: 0.4 },
    { id: "content", name: "Marketing de Contenidos", cost: 250, reach: 0.4, engagement: 0.9 }
  ]

  const simulateCampaign = () => {
    if (!selectedChannel || !campaignName || !targetAudience) {
      setFeedback(["Por favor completa todos los campos antes de simular"])
      return
    }

    setIsSimulating(true)
    const channel = channels.find(c => c.id === selectedChannel)
    
    if (!channel) return

    // Simular métricas basadas en las decisiones del usuario
    const newMetrics = {
      reach: Math.round(channel.reach * 100),
      engagement: Math.round(channel.engagement * 100),
      conversion: Math.round((channel.engagement * 0.3) * 100),
      roi: Math.round((channel.engagement * 0.5) * 100)
    }

    setMetrics(newMetrics)
    
    // Generar feedback basado en las métricas
    const newFeedback = [
      `Tu campaña "${campaignName}" ha alcanzado a ${newMetrics.reach}% de tu audiencia objetivo`,
      `El engagement es de ${newMetrics.engagement}%, lo que indica un buen nivel de interacción`,
      `La tasa de conversión es del ${newMetrics.conversion}%`,
      `El ROI estimado es del ${newMetrics.roi}%`
    ]

    setFeedback(newFeedback)
    setIsSimulating(false)
  }

  return (
    <div className="space-y-8">
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Constructor de Campañas de Marketing</h2>
        
        <div className="grid gap-6">
          <div className="space-y-2">
            <Label>Nombre de la Campaña</Label>
            <Input
              placeholder="Ej: Lanzamiento de Producto Q2"
              value={campaignName}
              onChange={(e) => setCampaignName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Audiencia Objetivo</Label>
            <Input
              placeholder="Ej: Profesionales 25-35 años interesados en tecnología"
              value={targetAudience}
              onChange={(e) => setTargetAudience(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Canal de Marketing</Label>
            <Select value={selectedChannel} onValueChange={setSelectedChannel}>
              <SelectTrigger>
                <SelectValue placeholder="Selecciona un canal" />
              </SelectTrigger>
              <SelectContent>
                {channels.map((channel) => (
                  <SelectItem key={channel.id} value={channel.id}>
                    {channel.name} - ${channel.cost}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Button 
            onClick={simulateCampaign}
            disabled={isSimulating}
            className="w-full"
          >
            {isSimulating ? "Simulando..." : "Simular Campaña"}
          </Button>
        </div>
      </Card>

      {feedback.length > 0 && (
        <Card className="p-6">
          <h3 className="text-xl font-semibold mb-4">Resultados de la Simulación</h3>
          
          <div className="grid gap-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Alcance</Label>
                <Progress value={metrics.reach} className="h-2" />
                <p className="text-sm text-muted-foreground">{metrics.reach}%</p>
              </div>
              <div className="space-y-2">
                <Label>Engagement</Label>
                <Progress value={metrics.engagement} className="h-2" />
                <p className="text-sm text-muted-foreground">{metrics.engagement}%</p>
              </div>
              <div className="space-y-2">
                <Label>Conversión</Label>
                <Progress value={metrics.conversion} className="h-2" />
                <p className="text-sm text-muted-foreground">{metrics.conversion}%</p>
              </div>
              <div className="space-y-2">
                <Label>ROI</Label>
                <Progress value={metrics.roi} className="h-2" />
                <p className="text-sm text-muted-foreground">{metrics.roi}%</p>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Análisis y Recomendaciones</Label>
              <div className="space-y-2">
                {feedback.map((item, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm">
                    <Badge variant="secondary">AI</Badge>
                    <p>{item}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}

      <Card className="p-6">
        <h3 className="text-xl font-semibold mb-4">Visualización de la Campaña</h3>
        <div id="stage-container" className="w-full bg-muted rounded-lg overflow-hidden">
          {stageSize.width > 0 && (
            <Stage width={stageSize.width} height={stageSize.height}>
              <Layer>
                <Group>
                  <Rect
                    x={stageSize.width * 0.05}
                    y={stageSize.height * 0.1}
                    width={stageSize.width * 0.9}
                    height={stageSize.height * 0.8}
                    fill="#f3f4f6"
                    stroke="#e5e7eb"
                    strokeWidth={2}
                  />
                  <Text
                    x={stageSize.width * 0.1}
                    y={stageSize.height * 0.3}
                    text={campaignName || "Nombre de la Campaña"}
                    fontSize={Math.min(stageSize.width * 0.03, 24)}
                    fill="#1f2937"
                  />
                  <Text
                    x={stageSize.width * 0.1}
                    y={stageSize.height * 0.5}
                    text={targetAudience || "Audiencia Objetivo"}
                    fontSize={Math.min(stageSize.width * 0.02, 16)}
                    fill="#4b5563"
                  />
                  {selectedChannel && (
                    <Circle
                      x={stageSize.width * 0.5}
                      y={stageSize.height * 0.5}
                      radius={Math.min(stageSize.width * 0.1, 50)}
                      fill="#3b82f6"
                      opacity={0.2}
                    />
                  )}
                </Group>
              </Layer>
            </Stage>
          )}
        </div>
      </Card>
    </div>
  )
} 