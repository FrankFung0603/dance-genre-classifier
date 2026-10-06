# 前端：Next.js（multi-stage：build 產出 → 精簡 runtime）
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# same-origin 部署：前端與 API 都走 nginx（80）→ NEXT_PUBLIC_API_BASE=""
# （瀏覽器呼叫 /session、WS /ws/* 由 nginx 反代到 backend）
ARG NEXT_PUBLIC_API_BASE=""
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app ./

EXPOSE 3000
CMD ["npx", "next", "start", "-p", "3000"]
