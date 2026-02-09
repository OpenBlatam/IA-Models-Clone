export interface CameraInfo {
  status: string;
  streaming: boolean;
  camera_index: number;
  resolution: {
    width: number;
    height: number;
  };
  fps: number;
  brightness: number;
  contrast: number;
  saturation: number;
  exposure: number;
}

export interface CameraSettings {
  camera_index?: number;
  resolution_width?: number;
  resolution_height?: number;
  fps?: number;
  brightness?: number;
  contrast?: number;
  saturation?: number;
  exposure?: number;
  auto_focus?: boolean;
  white_balance?: string;
}

