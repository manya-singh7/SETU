/// <reference types="vite/client" />

declare module "*.png" {
  const content: string;
  export default content;
}


declare module "node:fs" {
  export function readFileSync(path: string, encoding: string): string;
}
