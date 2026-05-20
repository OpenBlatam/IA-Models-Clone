import { z } from 'zod';

export const detectionSettingsSchema = z.object({
  confidence_threshold: z.number().min(0).max(1).optional(),
  nms_threshold: z.number().min(0).max(1).optional(),
  anomaly_threshold: z.number().min(0).max(1).optional(),
  use_autoencoder: z.boolean().optional(),
  use_statistical: z.boolean().optional(),
  object_detection_model: z.enum(['yolov8', 'faster_rcnn', 'ssd']).optional(),
  min_defect_size: z.number().min(1).optional(),
  max_defect_size: z.number().min(1).optional(),
});

export type DetectionSettingsForm = z.infer<typeof detectionSettingsSchema>;

