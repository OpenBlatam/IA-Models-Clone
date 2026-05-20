export interface BlogPost {
  id: string;
  date: string;
  category: string;
  title: string;
  ariaLabel: string;
  featured?: boolean;
  author?: string;
  content?: string;
  image?: string;
}

export interface PricingPlan {
  id: string;
  badge: string;
  title: string;
  price?: string;
  description: string;
  features?: string[];
  buttonText: string;
  buttonAriaLabel: string;
  comingSoon?: boolean;
}

export interface UseCase {
  id: string;
  title: string;
  description: string;
}

export interface Role {
  id: string;
  label: string;
  icon: string;
}

export interface Platform {
  id: string;
  name: string;
  downloads: {
    id: string;
    label: string;
    ariaLabel: string;
  }[];
}

export interface Feature {
  id: string;
  title: string;
  description: string;
}

