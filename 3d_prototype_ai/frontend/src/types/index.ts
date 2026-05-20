// Tipos principales basados en los schemas del backend

export enum ProductType {
  LICUADORA = 'licuadora',
  ESTUFA = 'estufa',
  MAQUINA = 'maquina',
  ELECTRODOMESTICO = 'electrodomestico',
  HERRAMIENTA = 'herramienta',
  MUEBLE = 'mueble',
  DISPOSITIVO = 'dispositivo',
  OTRO = 'otro',
}

export interface MaterialSource {
  name: string
  url?: string | null
  location?: string | null
  contact?: string | null
  availability: string
}

export interface Material {
  name: string
  quantity: number
  unit: string
  price_per_unit: number
  total_price: number
  category: string
  specifications?: Record<string, unknown> | null
  sources: MaterialSource[]
  alternatives: string[]
}

export interface CADPart {
  part_name: string
  part_number: number
  description: string
  material: string
  dimensions: Record<string, number>
  cad_file_path?: string | null
  cad_format: string
  quantity: number
}

export interface AssemblyStep {
  step_number: number
  description: string
  parts_involved: string[]
  tools_needed: string[]
  time_estimate?: string | null
  difficulty: string
  image_path?: string | null
}

export interface BudgetOption {
  budget_level: string
  total_cost: number
  materials: Material[]
  description: string
  trade_offs: string[]
  quality_level: string
}

export interface PrototypeRequest {
  product_description: string
  product_type?: ProductType | null
  budget?: number | null
  requirements?: string[]
  preferred_materials?: string[]
  location?: string | null
}

export interface PrototypeResponse {
  product_name: string
  product_description: string
  specifications: Record<string, unknown>
  materials: Material[]
  cad_parts: CADPart[]
  assembly_instructions: AssemblyStep[]
  budget_options: BudgetOption[]
  total_cost_estimate: number
  estimated_build_time: string
  difficulty_level: string
  generated_at: string
  documents: Record<string, string>
}

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface User {
  id: string
  email: string
  username: string
  role: string
  created_at: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export interface Notification {
  id: string
  type: string
  title: string
  message: string
  read: boolean
  created_at: string
}

export interface PrototypeHistory {
  id: string
  product_name: string
  product_type: ProductType
  total_cost_estimate: number
  generated_at: string
}

export interface AnalyticsData {
  total_prototypes: number
  total_cost: number
  average_cost: number
  most_common_type: ProductType
  trends: Array<{
    date: string
    count: number
    total_cost: number
  }>
}

export interface MarketplaceListing {
  id: string
  prototype_id: string
  title: string
  description: string
  price: number
  status: string
  created_at: string
  views: number
  likes: number
}

export interface GamificationStats {
  user_id: string
  points: number
  level: number
  achievements: Array<{
    id: string
    name: string
    description: string
    unlocked_at: string
  }>
  rank: number
}



