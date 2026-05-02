import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Bot } from "lucide-react"

interface AIBadgeProps {
  onClick?: () => void
}

export function AIBadge({ onClick }: AIBadgeProps) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={onClick}
          >
            <Bot className="h-4 w-4" />
            Powered by ChatGPT
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Este componente usa ChatGPT para generar contenido dinámico y educativo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
} 