/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    GITHUB_TOKEN: process.env.GITHUB_TOKEN,
    TRUTHGPT_API_PATH: process.env.TRUTHGPT_API_PATH || '../TruthGPT-main',
  },
}

module.exports = nextConfig


