'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import type { CameraSettings } from '../types';
import { cameraSettingsSchema } from '@/lib/validators/camera.validator';
import { DEFAULT_CAMERA_SETTINGS } from '@/config/constants';
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
import { Switch } from '@/components/ui/Switch';
import type { CameraSettingsForm } from '@/lib/validators/camera.validator';

interface CameraSettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (settings: CameraSettings) => Promise<void>;
}

const CameraSettingsModal = ({
  open,
  onClose,
  onSave,
}: CameraSettingsModalProps): JSX.Element => {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
    setValue,
  } = useForm<CameraSettingsForm>({
    resolver: zodResolver(cameraSettingsSchema),
    defaultValues: DEFAULT_CAMERA_SETTINGS,
  });

  const autoFocus = watch('auto_focus');

  const onSubmit = async (data: CameraSettingsForm): Promise<void> => {
    await onSave(data as CameraSettings);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Camera Settings</DialogTitle>
          <DialogDescription>
            Configure camera parameters for quality inspection
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="camera_index">Camera Index</Label>
              <Input
                id="camera_index"
                type="number"
                {...register('camera_index', { valueAsNumber: true })}
                error={errors.camera_index?.message}
                min="0"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="fps">FPS</Label>
              <Input
                id="fps"
                type="number"
                {...register('fps', { valueAsNumber: true })}
                error={errors.fps?.message}
                min="1"
                max="60"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="resolution_width">Resolution Width</Label>
              <Input
                id="resolution_width"
                type="number"
                {...register('resolution_width', { valueAsNumber: true })}
                error={errors.resolution_width?.message}
                min="320"
                max="7680"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="resolution_height">Resolution Height</Label>
              <Input
                id="resolution_height"
                type="number"
                {...register('resolution_height', { valueAsNumber: true })}
                error={errors.resolution_height?.message}
                min="240"
                max="4320"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="brightness">Brightness (0-1)</Label>
              <Input
                id="brightness"
                type="number"
                step="0.1"
                {...register('brightness', { valueAsNumber: true })}
                error={errors.brightness?.message}
                min="0"
                max="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="contrast">Contrast (0-1)</Label>
              <Input
                id="contrast"
                type="number"
                step="0.1"
                {...register('contrast', { valueAsNumber: true })}
                error={errors.contrast?.message}
                min="0"
                max="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="saturation">Saturation (0-1)</Label>
              <Input
                id="saturation"
                type="number"
                step="0.1"
                {...register('saturation', { valueAsNumber: true })}
                error={errors.saturation?.message}
                min="0"
                max="1"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="exposure">Exposure (0-1)</Label>
              <Input
                id="exposure"
                type="number"
                step="0.1"
                {...register('exposure', { valueAsNumber: true })}
                error={errors.exposure?.message}
                min="0"
                max="1"
              />
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="auto_focus"
              checked={autoFocus}
              onCheckedChange={(checked) => setValue('auto_focus', checked)}
            />
            <Label htmlFor="auto_focus" className="cursor-pointer">
              Auto Focus
            </Label>
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

export default CameraSettingsModal;
