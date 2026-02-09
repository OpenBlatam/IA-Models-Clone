import { memo, useState } from 'react';
import { useGeolocation } from '@/lib/hooks';
import Button from './Button';
import { MapPin, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface GeolocationButtonProps {
  onLocationFound?: (latitude: number, longitude: number) => void;
  className?: string;
  variant?: 'primary' | 'secondary' | 'danger';
}

const GeolocationButton = memo(({
  onLocationFound,
  className = '',
  variant = 'secondary',
}: GeolocationButtonProps): JSX.Element => {
  const { latitude, longitude, loading, error } = useGeolocation({ watch: false });
  const [requested, setRequested] = useState(false);

  const handleClick = (): void => {
    setRequested(true);
    if (latitude !== null && longitude !== null && onLocationFound) {
      onLocationFound(latitude, longitude);
    }
  };

  if (error) {
    return (
      <Button
        onClick={handleClick}
        variant={variant}
        disabled
        className={cn('flex items-center gap-2', className)}
        aria-label="Geolocation not available"
      >
        <MapPin className="w-4 h-4" />
        Location Unavailable
      </Button>
    );
  }

  return (
    <Button
      onClick={handleClick}
      variant={variant}
      isLoading={loading && requested}
      className={cn('flex items-center gap-2', className)}
      aria-label="Get current location"
    >
      {loading && requested ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          Getting Location...
        </>
      ) : (
        <>
          <MapPin className="w-4 h-4" />
          Get Location
        </>
      )}
    </Button>
  );
});

GeolocationButton.displayName = 'GeolocationButton';

export default GeolocationButton;



