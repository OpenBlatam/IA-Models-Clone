# Database Infrastructure (Prisma)

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![ORM](https://img.shields.io/badge/ORM-Prisma-blue.svg)
![Schema](https://img.shields.io/badge/schema-hardened-green.svg)

**Advanced database schema, migration tracking, and seeding logic for the Onyx Server core.**

</div>

---

## 📋 Overview

The **Prisma** module is the source of truth for the system's data architecture. It defines the relationships between users, AI models, production orders, and system settings. Leveraging the **Prisma ORM**, this module provides type-safe database access and automated schema synchronization across development and production environments.

## 📁 Structure

```
prisma/
├── schema.prisma         # The master data model definition
├── seed.ts               # Core system data and initial settings bootstrap
└── migrations/           # Version-controlled database schema changes
```

## 🚀 Tooling

| Command | Action |
|---------|-------------|
| `npx prisma db push` | Synchronize schema with the database. |
| `npx prisma generate` | Update the Prisma Client (types). |
| `npm run seed` | Populate the database with initial data. |

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
