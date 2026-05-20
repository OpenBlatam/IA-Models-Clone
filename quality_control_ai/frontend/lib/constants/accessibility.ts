export const ARIA_LABELS = {
  CAMERA: {
    VIEW: 'Camera view',
    FEED: 'Camera feed',
    START: 'Start inspection',
    STOP: 'Stop inspection',
    CAPTURE: 'Capture frame',
    SETTINGS: 'Camera settings',
  },
  INSPECTION: {
    RESULTS: 'Inspection results',
    UPLOAD: 'Upload image for inspection',
    FRAME: 'Inspect current frame',
  },
  ALERTS: {
    PANEL: 'Alerts panel',
    ITEM: 'Alert item',
  },
  STATISTICS: {
    PANEL: 'Statistics panel',
  },
  CONTROL: {
    PANEL: 'Control panel',
    DETECTION_SETTINGS: 'Detection settings',
  },
} as const;

export const KEYBOARD_SHORTCUTS = {
  START_INSPECTION: 's',
  STOP_INSPECTION: 'Escape',
  CAPTURE_FRAME: 'c',
  OPEN_SETTINGS: 's',
} as const;

