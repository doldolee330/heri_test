# vLLM 기동 전용 심(shim): handbook5 env의 flash_attn 패키지가 현재 torch와
# ABI 불일치(flash_attn_2_cuda undefined symbol)라 import가 깨진다.
# vLLM/transformers는 find_spec("flash_attn") 으로 사용 가능 여부를 판단하므로,
# 이 프로세스에서만 find_spec이 None을 반환하게 하여 네이티브 구현으로
# 폴백시킨다. (handbook5 env 자체는 변경하지 않음 — PYTHONPATH로만 주입)
import importlib.util

_orig_find_spec = importlib.util.find_spec


def _find_spec_no_flash_attn(name, *args, **kwargs):
    if name == "flash_attn" or name.startswith("flash_attn."):
        return None
    return _orig_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_no_flash_attn
