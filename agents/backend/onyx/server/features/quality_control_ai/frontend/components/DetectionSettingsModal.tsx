'use client';

import { useState } from 'react';
import { X } from 'lucide-react';
import type { DetectionSettings } from '@/lib/types';

interface DetectionSettingsModalProps {
  onClose: () => void;
  onSave: (settings: DetectionSettings) => Promise<void>;
}

const DetectionSettingsModal = ({
  onClose,
  onSave,
}: DetectionSettingsModalProps): JSX.Element => {
  const [settings, setSettings] = useState<DetectionSettings>({
    confidence_threshold: 0.5,
    nms_threshold: 0.4,
    anomaly_threshold: 0.7,
    use_autoencoder: true,
    use_statistical: true,
    object_detection_model: 'yolov8',
    min_defect_size: 10,
    max_defect_size: 10000,
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
            Detection Settings
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
                Confidence Threshold (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.confidence_threshold || 0.5}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    confidence_threshold: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                NMS Threshold (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.nms_threshold || 0.4}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    nms_threshold: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Anomaly Threshold (0-1)
              </label>
              <input
                type="number"
                step="0.1"
                value={settings.anomaly_threshold || 0.7}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    anomaly_threshold: parseFloat(e.target.value),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Detection Model
              </label>
              <select
                value={settings.object_detection_model || 'yolov8'}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    object_detection_model: e.target.value,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="yolov8">YOLOv8</option>
                <option value="faster_rcnn">Faster R-CNN</option>
                <option value="ssd">SSD</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Defect Size (px²)
              </label>
              <input
                type="number"
                value={settings.min_defect_size || 10}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    min_defect_size: parseInt(e.target.value, 10),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Defect Size (px²)
              </label>
              <input
                type="number"
                value={settings.max_defect_size || 10000}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    max_defect_size: parseInt(e.target.value, 10),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                min="1"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={settings.use_autoencoder || false}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    use_autoencoder: e.target.checked,
                  })
                }
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">Use Autoencoder</span>
            </label>

            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={settings.use_statistical || false}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    use_statistical: e.target.checked,
                  })
                }
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">
                Use Statistical Detection
              </span>
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

export default DetectionSettingsModal;

