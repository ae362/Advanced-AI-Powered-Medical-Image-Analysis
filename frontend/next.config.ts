import type { NextConfig } from "next";
const withNextIntl = require('next-intl/plugin')();
const nextConfig: NextConfig = {
  /* config options here */
};

module.exports = withNextIntl(nextConfig);
export default nextConfig;
