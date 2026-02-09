interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  fullScreen?: boolean;
  inline?: boolean;
}

export const Loading = ({ size = 'md', text, fullScreen = false, inline = false }: LoadingProps) => {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
  };

  const spinner = (
    <>
      <div
        className={`${sizes[size]} animate-spin rounded-full border-4 border-gray-200 border-t-primary-600`}
        role="status"
        aria-label="Cargando"
      />
      {text && <p className="text-sm text-gray-600">{text}</p>}
    </>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white bg-opacity-75">
        <div className="flex flex-col items-center justify-center gap-2">
          {spinner}
        </div>
      </div>
    );
  }

  if (inline) {
    return <span className="inline-flex items-center">{spinner}</span>;
  }

  return (
    <div className="flex flex-col items-center justify-center gap-2">
      {spinner}
    </div>
  );
};

