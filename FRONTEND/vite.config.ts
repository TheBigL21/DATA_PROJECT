import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import fs from "fs";

// Read backend port from .backend-port file (created by backend on startup)
function getBackendPort(): number {
  const portFile = path.resolve(__dirname, "../.backend-port");
  try {
    if (fs.existsSync(portFile)) {
      const port = parseInt(fs.readFileSync(portFile, "utf-8").trim(), 10);
      if (port > 0 && port < 65536) {
        return port;
      }
    }
  } catch (error) {
    console.warn("Could not read .backend-port file, using default port 5000");
  }
  return 5000; // Default fallback
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      "/api": {
        target: `http://localhost:${getBackendPort()}`,
        changeOrigin: true,
        secure: false,
      },
      "/health": {
        target: `http://localhost:${getBackendPort()}`,
        changeOrigin: true,
        secure: false,
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
