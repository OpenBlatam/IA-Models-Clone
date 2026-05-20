"use client"

import * as React from "react"
import { Bell } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"

export function NotificationBell() {
  const [notifications, setNotifications] = React.useState([
    {
      id: 1,
      title: "Nueva actualización disponible",
      time: "Hace 5 minutos",
    },
    {
      id: 2,
      title: "Tu perfil ha sido actualizado",
      time: "Hace 1 hora",
    },
  ])

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="relative size-8 px-0">
          <Bell className="size-5" />
          {notifications.length > 0 && (
            <span className="absolute -right-1 -top-1 flex size-4 items-center justify-center rounded-full bg-red-500 text-[10px] text-white">
              {notifications.length}
            </span>
          )}
          <span className="sr-only">Ver notificaciones</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        <div className="flex items-center justify-between px-4 py-2">
          <h4 className="text-sm font-medium">Notificaciones</h4>
          <Button
            variant="ghost"
            size="sm"
            className="h-auto px-2 text-xs text-muted-foreground"
            onClick={() => setNotifications([])}
          >
            Marcar todas como leídas
          </Button>
        </div>
        {notifications.length > 0 ? (
          notifications.map((notification) => (
            <DropdownMenuItem
              key={notification.id}
              className="flex flex-col items-start gap-1 px-4 py-3"
            >
              <p className="text-sm font-medium">{notification.title}</p>
              <p className="text-xs text-muted-foreground">{notification.time}</p>
            </DropdownMenuItem>
          ))
        ) : (
          <div className="flex items-center justify-center py-6 text-sm text-muted-foreground">
            No hay notificaciones
          </div>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
} 