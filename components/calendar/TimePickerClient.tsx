import dynamic from 'next/dynamic';
import type { TimePickerProps } from 'react-ios-time-picker';

export const TimePickerClient = dynamic<TimePickerProps>(
  () => import('react-ios-time-picker').then(mod => mod.TimePicker),
  { ssr: false }
); 