import { z } from 'zod';

export const cameraSettingsSchema = z.object({
  camera_index: z.number().min(0).optional(),
  resolution_width: z.number().min(320).max(7680).optional(),
  resolution_height: z.number().min(240).max(4320).optional(),
  fps: z.number().min(1).max(60).optional(),
  brightness: z.number().min(0).max(1).optional(),
  contrast: z.number().min(0).max(1).optional(),
  saturation: z.number().min(0).max(1).optional(),
  exposure: z.number().min(0).max(1).optional(),
  auto_focus: z.boolean().optional(),
  white_balance: z.string().optional(),
});

export type CameraSettingsForm = z.infer<typeof cameraSettingsSchema>;

