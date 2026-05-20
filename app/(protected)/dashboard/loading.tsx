import Skeleton from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";

const LoadingCard = () => (
  <div className="p-6 space-y-4 bg-white shadow-sm border border-gray-100 rounded-2xl">
    <div className="flex items-center space-x-4">
      <Skeleton circle height={48} width={48} />
      <div className="space-y-2">
        <Skeleton height={16} width={180} />
        <Skeleton height={16} width={120} />
      </div>
    </div>
    <Skeleton height={180} width="100%" style={{ borderRadius: 12 }} />
    <div className="flex items-center space-x-4">
      <Skeleton circle height={32} width={32} />
      <Skeleton circle height={32} width={32} />
      <Skeleton circle height={32} width={32} />
    </div>
  </div>
);

export default function DashboardLoading() {
  const skeletonCount = 6; // Ajusta según tu layout real

  return (
    <div className="min-h-screen bg-[#f8fafc] flex flex-col items-center justify-start py-10">
      <div className="w-full max-w-7xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
        {Array.from({ length: skeletonCount }).map((_, i) => (
          <LoadingCard key={i} />
        ))}
      </div>
    </div>
  );
}
