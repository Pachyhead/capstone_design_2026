// `react-router-dom` 자리에 들어가는 모듈. index.html 을 파일로 직접 열면(file://)
// 주소 경로를 쓰는 BrowserRouter 가 동작하지 않으므로, 같은 이름으로 HashRouter 를 내보낸다.
export * from 'react-router-dom';
export { HashRouter as BrowserRouter } from 'react-router-dom';
