// Auth Types
export interface CreateUserRequest {
  user_id: string;
  email?: string;
  name?: string;
  password?: string;
}

export interface UserResponse {
  user_id: string;
  email?: string;
  name?: string;
  created_at?: string;
  updated_at?: string;
}

export interface RegisterRequest {
  user_id: string;
  email?: string;
  password?: string;
  name?: string;
}

export interface RegisterResponse {
  user_id: string;
  email?: string;
  access_token: string;
  token_type: string;
  expires_in?: number;
}

export interface LoginRequest {
  user_id: string;
  password?: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  expires_in?: number;
}

export interface ProfileResponse {
  user_id: string;
  email?: string;
  name?: string;
  addiction_type?: string;
  days_sober?: number;
  created_at?: string;
  updated_at?: string;
}

export interface UpdateProfileRequest {
  email?: string;
  name?: string;
  addiction_type?: string;
  additional_info?: Record<string, any>;
}

