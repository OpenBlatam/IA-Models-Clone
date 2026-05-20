import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";

interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  isRead: boolean;
  createdAt: Date;
}

export function useNotifications() {
  const { data: session } = useSession();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (session?.user) {
      fetchNotifications();
    }
  }, [session]);

  const fetchNotifications = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch("/api/notifications");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setNotifications(Array.isArray(data) ? data : []);
    } catch (error: any) {
      setNotifications([]);
      setError(error?.message || "Error de red al obtener notificaciones");
    } finally {
      setIsLoading(false);
    }
  };

  const markAsRead = async (id: string) => {
    try {
      const response = await fetch("/api/notifications", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ id }),
      });

      if (response.ok) {
        setNotifications((prev) =>
          prev.map((notification) =>
            notification.id === id
              ? { ...notification, isRead: true }
              : notification
          )
        );
      }
    } catch (error) {
    }
  };

  return {
    notifications,
    isLoading,
    error,
    markAsRead,
    refetch: fetchNotifications,
  };
}  