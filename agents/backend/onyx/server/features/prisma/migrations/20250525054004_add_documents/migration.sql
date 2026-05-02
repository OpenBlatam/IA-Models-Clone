/*
  Warnings:

  - You are about to drop the column `url` on the `Document` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "Document" DROP COLUMN "url",
ADD COLUMN     "content" TEXT NOT NULL DEFAULT '',
ALTER COLUMN "type" SET DEFAULT 'text';
