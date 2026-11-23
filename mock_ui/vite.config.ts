import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';
import path from 'path';

const toneSrc = path.resolve(__dirname, '../tone/src');
const mockSrc = path.resolve(__dirname, 'src');

// 실제 Tone 화면 코드(tone/src)를 그대로 빌드하되, 바깥 세계와 닿는 모듈만
// mock 구현으로 바꿔 끼운다.
//   @/lib/api        → mockApi.ts   백엔드 호출을 시나리오대로 응답
//   @/data/mock      → mockData.ts  대화 상대 목록을 데모 프로필 3명으로
//   @/components/MetricsDock → noMetricsDock.tsx  시스템 지표 버튼을 띄우지 않음
//   react-router-dom → routerShim.ts  file:// 에서 열리도록 HashRouter 사용
const mock = (file: string) => path.join(mockSrc, file);

// react-router-dom 은 별칭으로 바꾸면 routerShim 자신이 원본을 불러올 수 없어서,
// 불러오는 쪽을 볼 수 있는 플러그인으로 처리한다.
function routerShim(): Plugin {
  const shim = mock('routerShim.ts');
  return {
    name: 'tone-mock-router-shim',
    enforce: 'pre',
    resolveId(source, importer) {
      if (source !== 'react-router-dom') return null;
      if (importer && path.normalize(importer) === shim) return null;
      return shim;
    },
  };
}

export default defineConfig({
  // index.html 을 더블클릭해 여는 것이 목표라 경로는 모두 상대 경로로 둔다.
  base: './',
  plugins: [routerShim(), react(), viteSingleFile()],
  resolve: {
    // 위에서부터 먼저 맞는 규칙이 적용되므로, 바꿔 끼울 모듈을 '@' 보다 앞에 둔다.
    // mockApi.ts 는 원본을 상대 경로로 불러오므로 이 규칙에 다시 걸리지 않는다.
    alias: [
      { find: /^@\/lib\/api$/, replacement: mock('mockApi.ts') },
      { find: /^@\/data\/mock$/, replacement: mock('mockData.ts') },
      { find: /^@\/components\/MetricsDock$/, replacement: mock('noMetricsDock.tsx') },
      { find: '@', replacement: toneSrc },
    ],
    // tone/src 의 파일도 이 폴더의 React 한 벌만 쓰게 한다(두 벌이면 훅이 깨진다).
    dedupe: ['react', 'react-dom', 'react-router-dom'],
  },
  server: {
    port: 3100,
    fs: { allow: [path.resolve(__dirname, '..')] },
  },
  // 음성은 src/audio 에서 불러와 HTML 안에 넣으므로, 따로 복사할 파일이 없다.
  publicDir: false,
  build: {
    // 저장소 루트 .gitignore 가 dist/ 를 무시하므로, 커밋해 둘 결과물은 site/ 에 둔다.
    outDir: 'site',
    emptyOutDir: true,
  },
});
