export const maskPhone = (phone: string): string => {
  const cleaned = phone.replace(/\D/g, '');
  if (cleaned.length <= 3) return cleaned;
  if (cleaned.length <= 6) {
    return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3)}`;
  }
  return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6, 10)}`;
};

export const maskCreditCard = (card: string, visibleDigits = 4): string => {
  const cleaned = card.replace(/\D/g, '');
  if (cleaned.length <= visibleDigits) return cleaned;
  const lastDigits = cleaned.slice(-visibleDigits);
  const masked = '*'.repeat(cleaned.length - visibleDigits);
  return `${masked}${lastDigits}`;
};

export const maskEmail = (email: string): string => {
  const [local, domain] = email.split('@');
  if (!domain) return email;

  const maskedLocal =
    local.length <= 2
      ? local
      : `${local[0]}${'*'.repeat(local.length - 2)}${local[local.length - 1]}`;

  return `${maskedLocal}@${domain}`;
};

export const maskSSN = (ssn: string): string => {
  const cleaned = ssn.replace(/\D/g, '');
  if (cleaned.length <= 3) return cleaned;
  if (cleaned.length <= 5) {
    return `${cleaned.slice(0, 3)}-${cleaned.slice(3)}`;
  }
  return `${cleaned.slice(0, 3)}-${cleaned.slice(3, 5)}-${cleaned.slice(5, 9)}`;
};

export const maskIP = (ip: string): string => {
  const parts = ip.split('.');
  if (parts.length !== 4) return ip;
  return `${parts[0]}.${parts[1]}.${'*'.repeat(parts[2].length)}.${parts[3]}`;
};

export const maskString = (
  str: string,
  startVisible = 0,
  endVisible = 0,
  maskChar = '*'
): string => {
  if (str.length <= startVisible + endVisible) return str;
  const start = str.slice(0, startVisible);
  const end = str.slice(-endVisible);
  const middle = maskChar.repeat(str.length - startVisible - endVisible);
  return `${start}${middle}${end}`;
};

