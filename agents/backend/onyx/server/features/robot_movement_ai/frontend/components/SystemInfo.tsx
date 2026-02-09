'use client';

import { useState } from 'react';
import { useTabs } from '@/lib/hooks/useTabs';
import { Info, Monitor, Cpu, HardDrive, MemoryStick, Globe } from 'lucide-react';

export default function SystemInfo() {
  const systemInfo = {
    os: 'Linux Ubuntu 22.04',
    arch: 'x86_64',
    nodeVersion: 'v18.17.0',
    browser: navigator.userAgent.split(' ')[0],
    cpu: 'Intel Core i7-12700K',
    memory: '32 GB',
    disk: '500 GB SSD',
    network: 'Ethernet 1 Gbps',
  };

  const { activeTab, setActiveTab } = useTabs(['general', 'hardware', 'network'], {
    initialTab: 'general',
  });

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Info className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Información del Sistema</h3>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          {(['general', 'hardware', 'network'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                activeTab === tab
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {tab === 'general' ? 'General' : tab === 'hardware' ? 'Hardware' : 'Red'}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="space-y-4">
          {activeTab === 'general' && (
            <>
              <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                <div className="flex items-center gap-2 mb-3">
                  <Monitor className="w-4 h-4 text-primary-400" />
                  <h4 className="font-semibold text-white">Sistema Operativo</h4>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">OS:</span>
                    <span className="text-white">{systemInfo.os}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Arquitectura:</span>
                    <span className="text-white">{systemInfo.arch}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Node.js:</span>
                    <span className="text-white">{systemInfo.nodeVersion}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Navegador:</span>
                    <span className="text-white">{systemInfo.browser}</span>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === 'hardware' && (
            <>
              <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                <div className="flex items-center gap-2 mb-3">
                  <Cpu className="w-4 h-4 text-primary-400" />
                  <h4 className="font-semibold text-white">Procesador</h4>
                </div>
                <p className="text-sm text-white">{systemInfo.cpu}</p>
              </div>
              <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                <div className="flex items-center gap-2 mb-3">
                  <MemoryStick className="w-4 h-4 text-primary-400" />
                  <h4 className="font-semibold text-white">Memoria</h4>
                </div>
                <p className="text-sm text-white">{systemInfo.memory}</p>
              </div>
              <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                <div className="flex items-center gap-2 mb-3">
                  <HardDrive className="w-4 h-4 text-primary-400" />
                  <h4 className="font-semibold text-white">Almacenamiento</h4>
                </div>
                <p className="text-sm text-white">{systemInfo.disk}</p>
              </div>
            </>
          )}

          {activeTab === 'network' && (
            <>
              <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                <div className="flex items-center gap-2 mb-3">
                  <Globe className="w-4 h-4 text-primary-400" />
                  <h4 className="font-semibold text-white">Red</h4>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Tipo:</span>
                    <span className="text-white">{systemInfo.network}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">IP Local:</span>
                    <span className="text-white">192.168.1.100</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Estado:</span>
                    <span className="text-green-400">Conectado</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
