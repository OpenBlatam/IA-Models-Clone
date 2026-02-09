'use client';

import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { useRobotStore } from '@/lib/store/robotStore';
import { useRecordingStore } from '@/lib/store/recordingStore';
import { useKeyboardShortcutsLegacy as useKeyboardShortcuts } from '@/lib/utils/keyboard';
import { handleApiError } from '@/lib/utils/error-handler';
import { Move, Square, Home, Navigation, Loader2 } from 'lucide-react';
import { Label } from '@/components/ui/Label';

const positionSchema = z.object({
  x: z.number().min(-10, 'X debe ser mayor a -10').max(10, 'X debe ser menor a 10'),
  y: z.number().min(-10, 'Y debe ser mayor a -10').max(10, 'Y debe ser menor a 10'),
  z: z.number().min(-10, 'Z debe ser mayor a -10').max(10, 'Z debe ser menor a 10'),
});

type PositionFormData = z.infer<typeof positionSchema>;

export default function RobotControl() {
  const { moveTo, stop, isLoading, currentPosition } = useRobotStore();
  const { isRecording, addRecord } = useRecordingStore();
  
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors },
    reset,
  } = useForm<PositionFormData>({
    resolver: zodResolver(positionSchema),
    defaultValues: {
      x: currentPosition?.x || 0,
      y: currentPosition?.y || 0,
      z: currentPosition?.z || 0,
    },
  });

  useEffect(() => {
    if (currentPosition) {
      reset({
        x: currentPosition.x,
        y: currentPosition.y,
        z: currentPosition.z,
      });
    }
  }, [currentPosition, reset]);

  const onSubmit = async (data: PositionFormData) => {
    try {
      const position = { x: data.x, y: data.y, z: data.z };
      await moveTo(position);
      if (isRecording) {
        addRecord(position, 'move');
      }
    } catch (error) {
      handleApiError(error, 'Error al mover el robot. Por favor, intenta de nuevo.');
    }
  };

  const handleStop = async () => {
    try {
      await stop();
      const currentPos = watch();
      if (isRecording) {
        addRecord({ x: currentPos.x, y: currentPos.y, z: currentPos.z }, 'stop');
      }
    } catch (error) {
      handleApiError(error, 'Error al detener el robot. Por favor, intenta de nuevo.');
    }
  };

  const handleHome = async () => {
    try {
      const homePos = { x: 0, y: 0, z: 0 };
      setValue('x', 0);
      setValue('y', 0);
      setValue('z', 0);
      await moveTo(homePos);
      if (isRecording) {
        addRecord(homePos, 'home');
      }
    } catch (error) {
      handleApiError(error, 'Error al mover a posición inicial. Por favor, intenta de nuevo.');
    }
  };

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 'h',
      ctrl: true,
      action: handleHome,
      description: 'Ir a posición home',
    },
    {
      key: 's',
      ctrl: true,
      action: handleStop,
      description: 'Detener robot',
    },
  ]);

  const presetPositions = [
    { name: 'Posición 1', x: 0.5, y: 0.3, z: 0.2 },
    { name: 'Posición 2', x: 0.3, y: 0.5, z: 0.4 },
    { name: 'Posición 3', x: 0.7, y: 0.2, z: 0.3 },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white rounded-lg p-tesla-lg md:p-tesla-xl border border-gray-200 shadow-sm"
    >
      <h2 className="text-2xl font-semibold text-tesla-black mb-tesla-lg md:mb-tesla-xl">Control del Robot</h2>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-tesla-lg md:space-y-tesla-xl">
        {/* Position Input */}
        <div>
          <h3 className="text-lg font-medium text-tesla-black mb-tesla-md md:mb-tesla-lg">Posición Objetivo</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-tesla-md">
            <div>
              <Label htmlFor="x" className="mb-tesla-sm">
                X (m)
              </Label>
              <input
                id="x"
                type="number"
                step="0.01"
                {...register('x', { valueAsNumber: true })}
                className={`w-full px-tesla-md py-tesla-sm bg-white border rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all ${
                  errors.x ? 'border-red-300' : 'border-gray-300'
                }`}
                aria-invalid={errors.x ? 'true' : 'false'}
                aria-describedby={errors.x ? 'x-error' : undefined}
              />
              {errors.x && (
                <p id="x-error" className="mt-tesla-xs text-sm text-red-600" role="alert">
                  {errors.x.message}
                </p>
              )}
            </div>
            <div>
              <Label htmlFor="y" className="mb-2">
                Y (m)
              </Label>
              <input
                id="y"
                type="number"
                step="0.01"
                {...register('y', { valueAsNumber: true })}
                className={`w-full px-tesla-md py-tesla-sm bg-white border rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all ${
                  errors.y ? 'border-red-300' : 'border-gray-300'
                }`}
                aria-invalid={errors.y ? 'true' : 'false'}
                aria-describedby={errors.y ? 'y-error' : undefined}
              />
              {errors.y && (
                <p id="y-error" className="mt-1 text-sm text-red-600" role="alert">
                  {errors.y.message}
                </p>
              )}
            </div>
            <div>
              <Label htmlFor="z" className="mb-2">
                Z (m)
              </Label>
              <input
                id="z"
                type="number"
                step="0.01"
                {...register('z', { valueAsNumber: true })}
                className={`w-full px-tesla-md py-tesla-sm bg-white border rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all ${
                  errors.z ? 'border-red-300' : 'border-gray-300'
                }`}
                aria-invalid={errors.z ? 'true' : 'false'}
                aria-describedby={errors.z ? 'z-error' : undefined}
              />
              {errors.z && (
                <p id="z-error" className="mt-1 text-sm text-red-600" role="alert">
                  {errors.z.message}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-tesla-sm md:gap-tesla-md">
          <motion.button
            type="submit"
            disabled={isLoading}
            whileHover={{ scale: isLoading ? 1 : 1.02 }}
            whileTap={{ scale: isLoading ? 1 : 0.98 }}
            className="flex-1 flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-tesla-blue hover:bg-opacity-90 text-white font-medium rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px]"
            aria-label="Mover robot a la posición especificada"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Move className="w-5 h-5" />
            )}
            Mover
          </motion.button>
          <motion.button
            type="button"
            onClick={handleStop}
            disabled={isLoading}
            whileHover={{ scale: isLoading ? 1 : 1.02 }}
            whileTap={{ scale: isLoading ? 1 : 0.98 }}
            className="flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black font-medium rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px]"
            aria-label="Detener movimiento del robot"
          >
            <Square className="w-5 h-5" />
            Detener
          </motion.button>
          <motion.button
            type="button"
            onClick={handleHome}
            disabled={isLoading}
            whileHover={{ scale: isLoading ? 1 : 1.02 }}
            whileTap={{ scale: isLoading ? 1 : 0.98 }}
            className="flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black font-medium rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px]"
            aria-label="Mover robot a posición home"
          >
            <Home className="w-5 h-5" />
            Home
          </motion.button>
        </div>
      </form>

      {/* Preset Positions */}
      <div className="mt-tesla-lg md:mt-tesla-xl">
        <h3 className="text-lg font-medium text-tesla-black mb-tesla-md md:mb-tesla-lg">Posiciones Predefinidas</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-tesla-sm md:gap-tesla-md">
          {presetPositions.map((preset, index) => (
            <motion.button
              key={index}
              type="button"
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                setValue('x', preset.x);
                setValue('y', preset.y);
                setValue('z', preset.z);
              }}
              className="p-tesla-md md:p-tesla-lg bg-white hover:bg-gray-50 rounded-md border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all text-left min-h-[44px]"
              aria-label={`Usar posición predefinida ${preset.name}`}
            >
              <Navigation className="w-5 h-5 text-tesla-blue mb-tesla-sm" />
              <p className="text-tesla-black font-medium text-sm">{preset.name}</p>
              <p className="text-tesla-gray-dark text-xs mt-tesla-xs font-mono">
                ({preset.x}, {preset.y}, {preset.z})
              </p>
            </motion.button>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

