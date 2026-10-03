# Stage 1: Build the React application with Vite
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package manifests and install dependencies
COPY package*.json ./
RUN npm ci || npm install

# Copy project source code and compile static build
COPY . .
RUN npm run build

# Stage 2: Production web server using lightweight Nginx
FROM nginx:alpine

# Copy compiled static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy Nginx config template with ${PORT} substitution support
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Set default port expected by Render container runtime
ENV PORT=10000
EXPOSE 10000

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
