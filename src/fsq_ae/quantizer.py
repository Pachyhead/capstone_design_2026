import torch
import torch.nn as nn
 
 
def round_with_ste(z):
    """
    forward는 round, backward는 identity로 그래디언트 통과
 
    Straight-Through Estimator: round 연산은 미분 불가능하지만,
    역전파 시에는 round를 무시하고 그래디언트를 그대로 흘려보낸다
    """
    return z + (z.round() - z).detach()
 
 
class FSQ(nn.Module):
    """
    Finite Scalar Quantization 모듈.
 
    Args:
        levels: 차원별 양자화 레벨 수 리스트 [5, 5, 5, 5, 5, 5, 5, 5]
                논문 권장 휴리스틱은 모든 i에 대해 L_i >= 5
    """
 
    def __init__(self, levels):
        super().__init__()
 
        # 학습 파라미터가 아니라 buffer로 등록 (state_dict에는 저장됨)
        levels_t = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", levels_t)
        self.register_buffer("half_width", levels_t // 2)
 
        self.num_dim = len(levels)
        self.codebook_size = int(torch.prod(levels_t).item())
 
    def forward(self, z):
        """
        연속 벡터 z를 양자화해서 [-1, 1]로 정규화된 z_q를 반환
 
        Args:
            z: (..., D) 연속 벡터 (인코더 출력)
 
        Returns:
            z_q: (..., D) 양자화된 벡터, 각 차원이 [-1, 1] 범위로 정규화됨
        """
        z_bounded = self._bound(z)
        z_quantized = round_with_ste(z_bounded)
        return z_quantized / self.half_width
    
    @torch.no_grad()
    def codes_to_indices(self, z_quantized_normalized):
        z_quantized = z_quantized_normalized * self.half_width
        indices = (z_quantized + self.half_width).long()
        return indices
    
    @torch.no_grad()
    def indices_to_codes(self, indices):
        z_quantized = indices.float() - self.half_width
        return z_quantized / self.half_width
 
    def _bound(self, z, eps=1e-3):
        """
        tanh를 이용해 z를 [-half_L, +half_L] 근방으로 압축
 
        even L의 경우 격자가 비대칭이라 offset/shift 보정
        """
        half_l = (self.levels - 1) * (1 - eps) / 2
 
        # odd L이면 offset=0, even L이면 offset=0.5 (격자 정렬용)
        offset = torch.where(
            self.levels % 2 == 1,
            torch.zeros_like(self.levels),
            torch.full_like(self.levels, 0.5),
        )
        shift = (offset / half_l).atanh()
 
        return (z + shift).tanh() * half_l - offset