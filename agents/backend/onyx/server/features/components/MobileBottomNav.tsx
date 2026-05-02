import Link from "next/link";
import { Home, Shield, Gift, Store, User } from "lucide-react";

export default function MobileBottomNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 bg-white border-t border-gray-200 flex justify-around items-center h-16 sm:hidden shadow-lg">
      <Link href="/dashboard" className="flex flex-col items-center justify-center text-blue-500">
        <span className="inline-block p-2 rounded-lg bg-blue-100"><Home className="w-7 h-7" /></span>
      </Link>
      <Link href="/dashboard/lessons" className="flex flex-col items-center justify-center text-blue-500">
        <span className="inline-block text-2xl">字</span>
      </Link>
      <Link href="/dashboard/achievements" className="flex flex-col items-center justify-center text-yellow-500">
        <span className="inline-block p-2"><Shield className="w-7 h-7" /></span>
      </Link>
      <Link href="/dashboard/rewards" className="flex flex-col items-center justify-center text-yellow-500">
        <span className="inline-block p-2"><Gift className="w-7 h-7" /></span>
      </Link>
      <Link href="/dashboard/shop" className="flex flex-col items-center justify-center text-red-500">
        <span className="inline-block p-2"><Store className="w-7 h-7" /></span>
      </Link>
      <Link href="/dashboard/profile" className="flex flex-col items-center justify-center text-purple-500">
        <span className="inline-block p-2"><User className="w-7 h-7" /></span>
      </Link>
    </nav>
  );
} 