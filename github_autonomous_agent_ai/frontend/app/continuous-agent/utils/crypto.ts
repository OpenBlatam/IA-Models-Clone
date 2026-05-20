export const randomString = (
  length: number = 16,
  charset: string = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
): string => {
  let result = "";
  for (let i = 0; i < length; i++) {
    result += charset.charAt(Math.floor(Math.random() * charset.length));
  }
  return result;
};

export const uuid = (): string => {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

export const generateId = (prefix?: string): string => {
  const id = randomString(16, "abcdefghijklmnopqrstuvwxyz0123456789");
  return prefix ? `${prefix}_${id}` : id;
};

export const hashString = async (str: string): Promise<string> => {
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
};

export const base64Encode = (str: string): string => {
  if (typeof window !== "undefined" && window.btoa) {
    return window.btoa(str);
  }
  return Buffer.from(str).toString("base64");
};

export const base64Decode = (str: string): string => {
  if (typeof window !== "undefined" && window.atob) {
    return window.atob(str);
  }
  return Buffer.from(str, "base64").toString();
};

export const encodeBase64Url = (str: string): string => {
  return base64Encode(str)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");
};

export const decodeBase64Url = (str: string): string => {
  let base64 = str.replace(/-/g, "+").replace(/_/g, "/");
  while (base64.length % 4) {
    base64 += "=";
  }
  return base64Decode(base64);
};

export const randomHex = (length: number = 16): string => {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
};

export const randomBytes = (length: number): Uint8Array => {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return bytes;
};





