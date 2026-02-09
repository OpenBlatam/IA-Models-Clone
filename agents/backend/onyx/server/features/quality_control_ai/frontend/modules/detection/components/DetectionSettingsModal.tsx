'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import type { DetectionSettings } from '../types';
import { detectionSettingsSchema } from '@/lib/validators/detection.validator';
import { DEFAULT_DETECTION_SETTINGS } from '@/config/constants';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/Dialog';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Label } from '@/components/ui/Label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select';
import { Switch } from '@/components/ui/Switch';
import type { DetectionSettingsForm } from '@/lib/validators/detection.validator';

interface DetectionSettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (settings: DetectionSettings) => Promise<void>;
}

const DetectionSettingsModal = ({
  open,
  onClose,
  onSave,
}: DetectionSettingsModalProps): JSX.Element => {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
    setValue,
  } = useForm<DetectionSettingsForm>({
    resolver: zodResolver(detectionSettingsSchema),
    defaultValues: DEFAULT_DETECTION_SETTINGS,
  });

  const useAutoencoder = watch('use_autoencoder');
  const useStatistical = watch('use_statistical');
  const detectionModel = watch('object_detection_model');

  const onSubmit = async (data: DetectionSettingsForm): Promise<void> => {
    await onSave(data as DetectionSettings);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Detection Settings</DialogTitle>
          <DialogDescription>
            Configure detection parameters and models
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="confidence_threshold">Confidence Threshold (0-1)</Label>
              <Input
                id="confidence_threshold"
                type="number"
                step="0.1"
                {...register('confidence_threshold', { valueAsNumber: true })}
                error={errors.confidence_threshold?.message}
                min="0"
                max="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="nms_threshold">NMS Threshold (0-1)</Label>
              <Input
                id="nms_threshold"
                type="number"
                step="0.1"
                {...register('nms_threshold', { valueAsNumber: true })}
                error={errors.nms_threshold?.message}
                min="0"
                max="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="anomaly_threshold">Anomaly Threshold (0-1)</Label>
              <Input
                id="anomaly_threshold"
                type="number"
                step="0.1"
                {...register('anomaly_threshold', { valueAsNumber: true })}
                error={errors.anomaly_threshold?.message}
                min="0"
                max="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="object_detection_model">Detection Model</Label>
              <Select
                value={detectionModel}
                onValueChange={(value) => setValue('object_detection_model', value as 'yolov8' | 'faster_rcnn' | 'ssd')}
              >
                <SelectTrigger id="object_detection_model">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="yolov8">YOLOv8</SelectItem>
                  <SelectItem value="faster_rcnn">Faster R-CNN</SelectItem>
                  <SelectItem value="ssd">SSD</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="min_defect_size">Min Defect Size (px²)</Label>
              <Input
                id="min_defect_size"
                type="number"
                {...register('min_defect_size', { valueAsNumber: true })}
                error={errors.min_defect_size?.message}
                min="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max_defect_size">Max Defect Size (px²)</Label>
              <Input
                id="max_defect_size"
                type="number"
                {...register('max_defect_size', { valueAsNumber: true })}
                error={errors.max_defect_size?.message}
                min="1"
              />
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Switch
                id="use_autoencoder"
                checked={useAutoencoder}
                onCheckedChange={(checked) => setValue('use_autoencoder', checked)}
              />
              <Label htmlFor="use_autoencoder" className="cursor-pointer">
                Use Autoencoder
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="use_statistical"
                checked={useStatistical}
                onCheckedChange={(checked) => setValue('use_statistical', checked)}
              />
              <Label htmlFor="use_statistical" className="cursor-pointer">
                Use Statistical Detection
              </Label>
            </div>
          </div>

          <DialogFooter>
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" isLoading={isSubmitting}>
              Save Settings
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default DetectionSettingsModal;
