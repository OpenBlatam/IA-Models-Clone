'use client';

import { useState } from 'react';
import { X } from 'lucide-react';
import type { CameraSettings } from '@/lib/types';

interface CameraSettingsModalProps {
  onClose: () => void;
  onSave: (settings: CameraSettings) => Promise<void>;
}

const CameraSettingsModal = ({
  onClose,
  onSave,
}: CameraSettingsModalProps): JSX.Element => {
  const [settings, setSettings] = useState<CameraSettings>({
    camera_index: 0,
    resolution_width: 1920,
    resolution_height: 1080,
    fps: 30,
    brightness: 0.5,
    contrast: 0.5,
    saturation: 0.5,
    exposure: 0.5,
    auto_focus: true,
    white_balance: 'auto',
  });

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    await onSave(settings);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-xl font-semibold text-gray-900">
            Camera Settings
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            tabIndex={0}
            aria-label="Close modal"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Camera Index
              </label>
              <input
                type="number"
                value={settings.camera_index || 0}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    camera_index: parseInt(e.target.value, 10),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                FPS
              </label>
              <input
                type="number"
                value={settings.fps || 30}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    fps: parseInt(e.target.value, 10),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="1"
                max="60"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Resolution Width
              </label>
              <input
                type="number"
                value={settings.resolution_width || 1920}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    resolution_width: parseInt(e.target.value, 10),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="320"
                max="7680"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Resolution Height
              </label>
              <input
                type="number"
                value={settings.resolution_height || 1080}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    resolution_height: parseInt(e.target.value, 10),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="240"
                max="4320"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Brightness (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.brightness || 0.5}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    brightness: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Contrast (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.contrast || 0.5}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    contrast: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Saturation (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.saturation || 0.5}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    saturation: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Exposure (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.exposure || 0.5}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    exposure: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={settings.auto_focus || false}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    auto_focus: e.target.checked,
                  })
                }
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">Auto Focus</span>
            </label>
          </div>

          <div className="flex items-center justify-end space-x-3 pt-4 border-t">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              Save Settings
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CameraSettingsModal;

