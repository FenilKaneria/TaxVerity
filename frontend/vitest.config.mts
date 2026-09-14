import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL(".", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    include: ["lib/**/*.test.ts", "lib/**/*.test.tsx"],
    // lib/config.ts reads this at import time (fail-fast, same as the
    // backend's Settings.require) — Next's own env loading doesn't apply
    // under vitest, so it has to be set here instead.
    env: {
      NEXT_PUBLIC_API_BASE_URL: "http://test.invalid",
    },
  },
});
