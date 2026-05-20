"use client";

import { Stage, Layer, Rect, Text } from "react-konva";
import { useState } from "react";

interface FigmaStageProps {
  selectedTool: string;
}

export default function FigmaStage({ selectedTool }: FigmaStageProps) {
  const [shapes, setShapes] = useState<any[]>([]);
  const [drawing, setDrawing] = useState(false);
  const [newRect, setNewRect] = useState<any>(null);

  // Manejar click en el canvas para agregar rectángulo o texto
  const handleStageClick = (e: any) => {
    if (selectedTool === "rect") {
      const stage = e.target.getStage();
      const pointer = stage.getPointerPosition();
      setShapes([
        ...shapes,
        {
          type: "rect",
          x: pointer.x,
          y: pointer.y,
          width: 120,
          height: 80,
          fill: "#6366f1",
          id: `rect-${shapes.length}`,
        },
      ]);
    } else if (selectedTool === "text") {
      const stage = e.target.getStage();
      const pointer = stage.getPointerPosition();
      setShapes([
        ...shapes,
        {
          type: "text",
          x: pointer.x,
          y: pointer.y,
          text: "Texto",
          fontSize: 24,
          fill: "#222",
          id: `text-${shapes.length}`,
        },
      ]);
    }
  };

  return (
    <Stage
      width={900}
      height={600}
      style={{ background: "#fff", cursor: selectedTool === "rect" ? "crosshair" : "default" }}
      onClick={handleStageClick}
    >
      <Layer>
        {shapes.map((shape) =>
          shape.type === "rect" ? (
            <Rect key={shape.id} {...shape} cornerRadius={8} shadowBlur={4} />
          ) : (
            <Text key={shape.id} {...shape} />
          )
        )}
      </Layer>
    </Stage>
  );
} 