'use client';

import { useState } from 'react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Grid, X, GripVertical } from 'lucide-react';
import { motion } from 'framer-motion';
import { useRobotStore } from '@/lib/store/robotStore';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface Widget {
  id: string;
  type: 'status' | 'metrics' | 'chart' | 'position';
  title: string;
  x: number;
  y: number;
  w: number;
  h: number;
  data?: any;
}

const defaultWidgets: Widget[] = [
  { id: '1', type: 'status', title: 'Estado del Robot', x: 0, y: 0, w: 4, h: 2 },
  { id: '2', type: 'position', title: 'Posición Actual', x: 4, y: 0, w: 4, h: 2 },
  { id: '3', type: 'metrics', title: 'Métricas Rápidas', x: 8, y: 0, w: 4, h: 2 },
  { id: '4', type: 'chart', title: 'Gráfico de Movimiento', x: 0, y: 2, w: 12, h: 4 },
];

function SortableWidget({ widget, isEditing, onDelete, renderWidget }: {
  widget: Widget;
  isEditing: boolean;
  onDelete: () => void;
  renderWidget: (widget: Widget) => React.ReactNode;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: widget.id });

  const dragStyle = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={{
        gridColumn: `span ${widget.w}`,
        gridRow: `span ${widget.h}`,
        ...dragStyle,
      }}
      className={`bg-white rounded-lg border border-gray-200 shadow-sm relative transition-all hover:shadow-tesla-md ${
        isEditing ? 'cursor-move ring-2 ring-tesla-blue/20' : ''
      }`}
    >
      {isEditing && (
        <div className="absolute top-2 right-2 flex gap-1 z-10">
          <button
            {...attributes}
            {...listeners}
            className="p-2 bg-white border border-gray-300 hover:border-gray-400 rounded-md transition-all min-h-[44px] min-w-[44px] flex items-center justify-center cursor-grab active:cursor-grabbing"
            aria-label="Mover widget"
          >
            <GripVertical className="w-4 h-4 text-tesla-gray-dark" />
          </button>
          <button
            onClick={onDelete}
            className="p-2 bg-red-600 hover:bg-red-700 rounded-md transition-all min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Eliminar widget"
          >
            <X className="w-4 h-4 text-white" />
          </button>
        </div>
      )}
      <div className="p-3 border-b border-gray-200">
        <h4 className="text-sm font-semibold text-tesla-black">{widget.title}</h4>
      </div>
      <div className="h-full overflow-hidden">{renderWidget(widget)}</div>
    </div>
  );
}

export default function WidgetDashboard() {
  const { status } = useRobotStore();
  const [widgets, setWidgets] = useState<Widget[]>(defaultWidgets);
  const [isEditing, setIsEditing] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      setWidgets((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  const renderWidget = (widget: Widget) => {
    switch (widget.type) {
      case 'status':
        return (
          <div className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-tesla-gray-dark text-sm font-medium">Conexión</span>
                <span
                  className={`px-3 py-1 rounded-md text-xs font-semibold ${
                    status?.robot_status.connected
                      ? 'bg-green-50 text-green-700 border border-green-200'
                      : 'bg-red-50 text-red-700 border border-red-200'
                  }`}
                >
                  {status?.robot_status.connected ? 'Conectado' : 'Desconectado'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-tesla-gray-dark text-sm font-medium">Movimiento</span>
                <span
                  className={`px-3 py-1 rounded-md text-xs font-semibold ${
                    status?.robot_status.moving
                      ? 'bg-yellow-50 text-yellow-700 border border-yellow-200'
                      : 'bg-gray-50 text-tesla-gray-dark border border-gray-200'
                  }`}
                >
                  {status?.robot_status.moving ? 'En Movimiento' : 'Detenido'}
                </span>
              </div>
            </div>
          </div>
        );

      case 'position':
        return (
          <div className="p-4">
            {status?.robot_status.position ? (
              <div className="space-y-2">
                <div className="text-sm">
                  <span className="text-tesla-gray-dark font-medium">X: </span>
                  <span className="text-tesla-black font-semibold font-mono">
                    {status.robot_status.position.x.toFixed(3)}m
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-tesla-gray-dark font-medium">Y: </span>
                  <span className="text-tesla-black font-semibold font-mono">
                    {status.robot_status.position.y.toFixed(3)}m
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-tesla-gray-dark font-medium">Z: </span>
                  <span className="text-tesla-black font-semibold font-mono">
                    {status.robot_status.position.z.toFixed(3)}m
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-tesla-gray-dark text-sm">No hay posición disponible</p>
            )}
          </div>
        );

      case 'metrics':
        return (
          <div className="p-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-gray-50 rounded-md border border-gray-200">
                <p className="text-xs text-tesla-gray-dark mb-1 font-medium">CPU</p>
                <p className="text-lg font-bold text-tesla-black">45%</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-md border border-gray-200">
                <p className="text-xs text-tesla-gray-dark mb-1 font-medium">Memoria</p>
                <p className="text-lg font-bold text-tesla-black">62%</p>
              </div>
            </div>
          </div>
        );

      case 'chart':
        const chartData = Array.from({ length: 20 }, (_, i) => ({
          name: `T${i}`,
          value: Math.random() * 100,
        }));
        return (
          <div className="p-4 h-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" stroke="#393c41" />
                <YAxis stroke="#393c41" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px', color: '#171a20' }}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#0062cc"
                  fill="#0062cc"
                  fillOpacity={0.2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Grid className="w-5 h-5 text-tesla-blue" />
          <h3 className="text-lg font-semibold text-tesla-black">Dashboard Personalizable</h3>
        </div>
        <button
          onClick={() => setIsEditing(!isEditing)}
          className="px-5 py-2 bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium min-h-[44px]"
          aria-label={isEditing ? 'Guardar cambios' : 'Editar dashboard'}
        >
          {isEditing ? 'Guardar' : 'Editar'}
        </button>
      </div>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <SortableContext items={widgets.map((w) => w.id)}>
          <div className="grid grid-cols-12 gap-4 auto-rows-[100px]">
            {widgets.map((widget) => (
              <SortableWidget
                key={widget.id}
                widget={widget}
                isEditing={isEditing}
                onDelete={() => setWidgets(widgets.filter((w) => w.id !== widget.id))}
                renderWidget={renderWidget}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>
    </div>
  );
}

