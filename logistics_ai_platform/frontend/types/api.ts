export type TransportationMode = 'air' | 'maritime' | 'ground' | 'multimodal';

export type ShipmentStatus =
  | 'pending'
  | 'quoted'
  | 'booked'
  | 'in_transit'
  | 'in_customs'
  | 'delivered'
  | 'delayed'
  | 'cancelled'
  | 'exception';

export type ContainerStatus =
  | 'empty'
  | 'loaded'
  | 'in_transit'
  | 'at_port'
  | 'at_terminal'
  | 'delivered'
  | 'damaged';

export interface Location {
  country: string;
  city: string;
  port_code?: string;
  address?: string;
  latitude?: number;
  longitude?: number;
  timezone?: string;
}

export interface GPSLocation {
  latitude: number;
  longitude: number;
  timestamp: string;
  accuracy?: number;
  speed?: number;
  heading?: number;
}

export interface CargoDetails {
  description: string;
  weight_kg: number;
  volume_m3?: number;
  quantity: number;
  unit_type: string;
  value_usd?: number;
  hs_code?: string;
  dangerous_goods: boolean;
  temperature_controlled: boolean;
  special_handling?: string;
}

export interface QuoteRequest {
  origin: Location;
  destination: Location;
  cargo: CargoDetails;
  transportation_mode: TransportationMode;
  preferred_departure_date?: string;
  preferred_arrival_date?: string;
  insurance_required: boolean;
  special_requirements?: string;
}

export interface QuoteOption {
  quote_id: string;
  transportation_mode: TransportationMode;
  carrier: string;
  estimated_departure: string;
  estimated_arrival: string;
  transit_days: number;
  price_usd: number;
  currency: string;
  service_level: string;
  features: string[];
}

export interface QuoteResponse {
  quote_id: string;
  request_id: string;
  origin: Location;
  destination: Location;
  cargo: CargoDetails;
  options: QuoteOption[];
  valid_until: string;
  created_at: string;
}

export interface BookingRequest {
  quote_id: string;
  selected_option_id: string;
  shipper_info: Record<string, any>;
  consignee_info: Record<string, any>;
  payment_terms: string;
  special_instructions?: string;
}

export interface BookingResponse {
  booking_id: string;
  quote_id: string;
  shipment_id: string;
  status: ShipmentStatus;
  booking_reference: string;
  carrier_reference?: string;
  created_at: string;
  estimated_departure: string;
  estimated_arrival: string;
}

export interface ShipmentRequest {
  booking_id?: string;
  origin: Location;
  destination: Location;
  cargo: CargoDetails;
  transportation_mode: TransportationMode;
  carrier?: string;
}

export interface TrackingEvent {
  event_type: string;
  location: Location;
  timestamp: string;
  description: string;
  status: ShipmentStatus;
  metadata: Record<string, any>;
}

export interface ShipmentResponse {
  shipment_id: string;
  booking_id?: string;
  shipment_reference: string;
  tracking_number?: string;
  house_bill_number?: string;
  master_bill_number?: string;
  origin: Location;
  destination: Location;
  cargo: CargoDetails;
  transportation_mode: TransportationMode;
  status: ShipmentStatus;
  carrier?: string;
  tracking_events: TrackingEvent[];
  estimated_departure?: string;
  estimated_arrival?: string;
  actual_departure?: string;
  actual_arrival?: string;
  created_at: string;
  updated_at: string;
}

export interface ContainerRequest {
  container_type: string;
  shipment_id?: string;
  container_number?: string;
}

export interface ContainerResponse {
  container_id: string;
  container_number: string;
  container_type: string;
  shipment_id?: string;
  status: ContainerStatus;
  location?: Location;
  gps_location?: GPSLocation;
  loaded_at?: string;
  sealed_at?: string;
  created_at: string;
  updated_at: string;
}

export interface InvoiceRequest {
  shipment_id: string;
  invoice_number?: string;
  due_date?: string;
  notes?: string;
}

export interface InvoiceLineItem {
  description: string;
  quantity: number;
  unit_price: number;
  total: number;
  tax_rate?: number;
}

export interface InvoiceResponse {
  invoice_id: string;
  invoice_number: string;
  shipment_id: string;
  issue_date: string;
  due_date?: string;
  subtotal: number;
  tax: number;
  total: number;
  currency: string;
  line_items: InvoiceLineItem[];
  status: string;
  pdf_url?: string;
  created_at: string;
}

export interface DocumentRequest {
  shipment_id: string;
  document_type: string;
  file_name: string;
  description?: string;
}

export interface DocumentResponse {
  document_id: string;
  shipment_id: string;
  document_type: string;
  file_name: string;
  file_url: string;
  file_size: number;
  mime_type: string;
  description?: string;
  uploaded_at: string;
}

export interface AlertResponse {
  alert_id: string;
  shipment_id?: string;
  container_id?: string;
  alert_type: string;
  message: string;
  priority: string;
  is_read: boolean;
  created_at: string;
  read_at?: string;
}

export interface InsuranceRequest {
  shipment_id: string;
  coverage_type: string;
  coverage_amount_usd: number;
  deductible_usd?: number;
}

export interface InsuranceResponse {
  insurance_id: string;
  shipment_id: string;
  coverage_type: string;
  coverage_amount_usd: number;
  premium_usd: number;
  deductible_usd: number;
  policy_number: string;
  status: string;
  valid_from: string;
  valid_until: string;
  created_at: string;
}




