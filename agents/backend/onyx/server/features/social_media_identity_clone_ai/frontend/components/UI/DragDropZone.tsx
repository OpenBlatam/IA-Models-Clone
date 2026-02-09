import { memo, useState, useCallback } from 'react';
import { useDragAndDrop } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Upload, File } from 'lucide-react';

interface DragDropZoneProps {
  onDrop: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  maxSize?: number;
  className?: string;
  children?: React.ReactNode;
}

const DragDropZone = memo(({
  onDrop,
  accept,
  multiple = false,
  maxSize,
  className = '',
  children,
}: DragDropZoneProps): JSX.Element => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      setError(null);

      const files = Array.from(e.dataTransfer.files);

      if (accept) {
        const acceptedTypes = accept.split(',').map((type) => type.trim());
        const invalidFiles = files.filter(
          (file) => !acceptedTypes.some((type) => file.type.match(type.replace('*', '.*')))
        );

        if (invalidFiles.length > 0) {
          setError(`Invalid file type. Accepted: ${accept}`);
          return;
        }
      }

      if (maxSize) {
        const oversizedFiles = files.filter((file) => file.size > maxSize);
        if (oversizedFiles.length > 0) {
          setError(`File size exceeds ${maxSize / 1024 / 1024}MB`);
          return;
        }
      }

      if (!multiple && files.length > 1) {
        setError('Only one file is allowed');
        return;
      }

      onDrop(files);
    },
    [onDrop, accept, multiple, maxSize]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        onDrop(files);
      }
    },
    [onDrop]
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={cn(
        'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
        isDragging ? 'border-primary-500 bg-primary-50' : 'border-gray-300',
        className
      )}
    >
      <input
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleFileInput}
        className="hidden"
        id="file-input"
      />
      <label htmlFor="file-input" className="cursor-pointer">
        {children || (
          <>
            <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-600 mb-2">
              Drag and drop files here, or click to select
            </p>
            <p className="text-sm text-gray-500">
              {accept && `Accepted: ${accept}`}
              {maxSize && ` • Max size: ${maxSize / 1024 / 1024}MB`}
            </p>
          </>
        )}
      </label>

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-600">
          {error}
        </div>
      )}
    </div>
  );
});

DragDropZone.displayName = 'DragDropZone';

export default DragDropZone;



