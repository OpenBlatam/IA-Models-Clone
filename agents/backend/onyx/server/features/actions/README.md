# Server Actions

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Type](https://img.shields.io/badge/type-nextjs--server--actions-blue.svg)
![Env](https://img.shields.io/badge/environment-typescript-blue.svg)

**Secure, server-side asynchronous functions for handling user operations and third-party integrations.**

</div>

---

## 📋 Overview

The **Actions** module contains Next.js Server Actions designed for secure data mutation and state management. These actions bridge the frontend UI with backend services like Stripe and the primary database, ensuring that sensitive logic (like role updates or billing portal generation) remains strictly server-side.

## 🚀 Key Actions

| Action | Description |
|---------|-------------|
| `generate-user-stripe.ts` | Creates and manages Stripe customer sessions for users. |
| `open-customer-portal.ts` | Generates secure links to the Stripe billing management portal. |
| `update-user-name.ts` | Handles profile name updates with validation. |
| `update-user-role.ts` | Securely manages user authorization levels and roles. |

## 💻 Usage

```typescript
"use server";

import { updateUserName } from "@/features/actions/update-user-name";

// Integrating into a React component
async function ActionHandler(formData: FormData) {
  const result = await updateUserName(formData);
  // Handle success/error
}
```

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
