import { useRef, useCallback, useState } from 'react';
import { cn } from '@/lib/utils';
import Button from './Button';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number;
  className?: string;
  label?: string;
  disabled?: boolean;
}

const FileUpload = ({
  onFileSelect,
  accept,
  maxSize,
  className = '',
  label = 'Upload File',
  disabled = false,
}: FileUploadProps): JSX.Element => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>): void => {
      const file = e.target.files?.[0];
      if (!file) {
        return;
      }

      if (maxSize && file.size > maxSize) {
        setError(`File size must be less than ${(maxSize / 1024 / 1024).toFixed(2)}MB`);
        return;
      }

      setError(null);
      onFileSelect(file);
    },
    [onFileSelect, maxSize]
  );

  const handleClick = useCallback((): void => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [disabled]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>): void => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleClick();
      }
    },
    [handleClick]
  );

  return (
    <div className={cn('space-y-2', className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
        aria-label={label}
      />
      <Button
        type="button"
        variant="secondary"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className="w-full"
      >
        {label}
      </Button>
      {error && (
        <p className="text-sm text-red-600" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export default FileUpload;



