import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 專案在 home 目錄之下：明確 root，避免上層 lockfile 干擾（turbopack 警告）
  turbopack: { root: process.cwd() },
};

export default nextConfig;
