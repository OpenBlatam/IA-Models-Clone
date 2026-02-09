import { QRCodeSVG } from 'qrcode.react';
import { cn } from '@/lib/utils';

interface QRCodeProps {
  value: string;
  size?: number;
  level?: 'L' | 'M' | 'Q' | 'H';
  className?: string;
}

const QRCode = ({ value, size = 200, level = 'M', className = '' }: QRCodeProps): JSX.Element => {
  return (
    <div className={cn('flex items-center justify-center', className)}>
      <QRCodeSVG value={value} size={size} level={level} />
    </div>
  );
};

export default QRCode;



