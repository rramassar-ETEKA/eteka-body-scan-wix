/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          // Allow embedding in Wix iframes
          { key: "X-Frame-Options", value: "ALLOWALL" },
          {
            key: "Content-Security-Policy",
            value: "frame-ancestors *;",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
