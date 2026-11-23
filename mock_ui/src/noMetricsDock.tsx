// `@/components/MetricsDock` 자리에 들어가는 모듈. 데모에서는 시스템 지표 버튼을 띄우지 않는다.
//
// 지표 패널은 실제 서버 호출의 응답 시간과 전송 크기를 보여 주는 화면인데, 여기서는
// 서버가 없어서 응답 시간이 mockApi 가 일부러 넣은 대기 시간일 뿐이다. 그 숫자를
// 측정값처럼 보여 주면 오해를 사므로 통째로 뺀다.
export function MetricsDock(_props: { variant?: 'light' | 'dark' }) {
  return null;
}
