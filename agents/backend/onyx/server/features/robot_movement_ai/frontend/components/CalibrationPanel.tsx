'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { useRobotStore } from '@/lib/store/robotStore';
import { Target, RotateCcw, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function CalibrationPanel() {
  const { status } = useRobotStore();
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState<'idle' | 'calibrating' | 'success' | 'error'>('idle');

  const handleCalibrate = async () => {
    setIsCalibrating(true);
    setCalibrationStatus('calibrating');
    try {
      // Simulate calibration
      await new Promise((resolve) => setTimeout(resolve, 3000));
      setCalibrationStatus('success');
      toast.success('Calibración completada exitosamente');
    } catch (error: any) {
      setCalibrationStatus('error');
      toast.error(`Error: ${error.message || 'Failed to calibrate'}`);
    } finally {
      setIsCalibrating(false);
    }
  };

  const handleReset = async () => {
    try {
      await apiClient.moveTo({ x: 0, y: 0, z: 0 });
      toast.success('Robot reiniciado a posición inicial');
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to reset'}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Target className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Calibración del Robot</h3>
        </div>

        {/* Current Status */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-semibold text-white mb-3">Estado Actual</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-300">Posición X:</span>
              <span className="text-white font-mono">
                {status?.robot_status.position?.x.toFixed(3) || '0.000'} m
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Posición Y:</span>
              <span className="text-white font-mono">
                {status?.robot_status.position?.y.toFixed(3) || '0.000'} m
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Posición Z:</span>
              <span className="text-white font-mono">
                {status?.robot_status.position?.z.toFixed(3) || '0.000'} m
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Estado:</span>
              <span className={`font-semibold ${
                status?.robot_status.connected ? 'text-green-400' : 'text-red-400'
              }`}>
                {status?.robot_status.connected ? 'Conectado' : 'Desconectado'}
              </span>
            </div>
          </div>
        </div>

        {/* Calibration Status */}
        {calibrationStatus !== 'idle' && (
          <div className={`mb-6 p-4 rounded-lg border ${
            calibrationStatus === 'calibrating'
              ? 'bg-yellow-500/10 border-yellow-500/50'
              : calibrationStatus === 'success'
              ? 'bg-green-500/10 border-green-500/50'
              : 'bg-red-500/10 border-red-500/50'
          }`}>
            <div className="flex items-center gap-2">
              {calibrationStatus === 'calibrating' && (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-400"></div>
                  <span className="text-yellow-400">Calibrando...</span>
                </>
              )}
              {calibrationStatus === 'success' && (
                <>
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-green-400">Calibración exitosa</span>
                </>
              )}
              {calibrationStatus === 'error' && (
                <>
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <span className="text-red-400">Error en calibración</span>
                </>
              )}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="space-y-3">
          <button
            onClick={handleCalibrate}
            disabled={isCalibrating || !status?.robot_status.connected}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Target className="w-5 h-5" />
            {isCalibrating ? 'Calibrando...' : 'Iniciar Calibración'}
          </button>
          <button
            onClick={handleReset}
            disabled={!status?.robot_status.connected}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RotateCcw className="w-5 h-5" />
            Reiniciar a Posición Inicial
          </button>
        </div>

        {/* Instructions */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-semibold text-white mb-2">Instrucciones:</h4>
          <ul className="text-xs text-gray-300 space-y-1">
            <li>• Asegúrate de que el robot esté conectado</li>
            <li>• El robot debe estar en una posición segura</li>
            <li>• La calibración puede tardar varios segundos</li>
            <li>• No muevas el robot durante la calibración</li>
          </ul>
        </div>
      </div>
    </div>
  );
}


