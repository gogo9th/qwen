from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch
import argparse
import time

# 예시:
# python3 run_qwen.py --param 0.6  -> Qwen/Qwen3-0.6B-Instruct
# python3 run_qwen.py --param 4    -> Qwen/Qwen3-4B
parser = argparse.ArgumentParser()
parser.add_argument(
    "--param",
    default="0.6",
    help="모델 크기 파라미터: 0.6, 4, 8 (기본값: 0.6)"
)
parser.add_argument(
    "--bits",
    default="4",
    choices=["4", "16"],
    help="가중치 정밀도 비트 선택: 4, 16 (기본값: 4)"
)
parser.add_argument(
    "--no-stream",
    action="store_true",
    help="스트리밍 출력 비활성화 (생성 완료 후 한꺼번에 출력)"
)
args = parser.parse_args()

model_map = {
    "0.6": "Qwen/Qwen3-0.6B",
    "4": "Qwen/Qwen3-4B",
    "8": "Qwen/Qwen3-8B",
}

if args.param not in model_map:
    raise ValueError(
        f"지원하지 않는 --param 값: {args.param}. 지원값: {', '.join(model_map.keys())}"
    )

model_name = model_map[args.param]

print(f"모델 다운로드 및 로딩 중: {model_name} ...")
print(f"정밀도 설정: {args.bits}-bit")

# 1. 모델과 토크나이저 로드
# device_map="auto": GPU가 있으면 GPU로, 없으면 CPU로 자동 할당
if args.bits == "4":
    # 4-bit 양자화 로딩
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config
    )
else:
    # 16-bit 계열(float16/bfloat16)을 자동 선택
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 실제 모델이 올라간 장치를 출력 (GPU/CPU)
model_device = next(model.parameters()).device
if model_device.type == "cuda":
    print(f"실행 장치: GPU ({model_device})")
else:
    print(f"실행 장치: CPU ({model_device})")

# 모델 정보 출력
config = model.config
context_window = getattr(config, "max_position_embeddings", None)
if context_window is None:
    context_window = getattr(config, "seq_length", None)
if context_window is None:
    context_window = getattr(config, "n_positions", "unknown")

total_params = sum(p.numel() for p in model.parameters())
embedding_size = getattr(config, "hidden_size", None)
if embedding_size is None:
    embedding_size = getattr(config, "n_embd", None)
if embedding_size is None:
    embedding_size = getattr(config, "d_model", "unknown")

print(f"모델 정보:")
print(f"- model_name: {model_name}")
print(f"- context_window: {context_window}")
print(f"- embedding_size: {embedding_size}")
print(f"- parameters: {total_params:,}")
print(f"- vocab_size: {getattr(config, 'vocab_size', 'unknown')}")

# 2. 대화 메시지 구성
prompt = "인공지능의 미래에 대해 한 문장으로 설명해줘."
messages = [
    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
    {"role": "user", "content": prompt}
]

# 3. 입력 데이터 전처리 (Chat Template 적용)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. 답변 생성 (Inference)
print("답변 생성 중...")
print("-" * 20)
print(f"질문: {prompt}")
print("답변: ", end="", flush=True)

start_time = time.perf_counter()
if args.no_stream:
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=512,  # 최대 출력 길이
        temperature=0.7,     # 창의성 조절 (0.0 ~ 1.0)
        top_p=0.9,           # 확률 분포 조절
    )
else:
    # 토큰이 생성될 때마다 즉시 터미널에 출력
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=512,  # 최대 출력 길이
        temperature=0.7,     # 창의성 조절 (0.0 ~ 1.0)
        top_p=0.9,           # 확률 분포 조절
        streamer=streamer
    )
end_time = time.perf_counter()

input_token_count = model_inputs["input_ids"].shape[-1]
output_token_count = outputs.shape[-1]
generated_token_count = max(output_token_count - input_token_count, 0)
elapsed_time = end_time - start_time
tokens_per_sec = (
    generated_token_count / elapsed_time if elapsed_time > 0 else float("inf")
)

if args.no_stream:
    generated_ids = outputs[0, input_token_count:]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(decoded_text)

# 5. 마무리 출력
print()
print("-" * 20)
print(f"생성 토큰 수: {generated_token_count}")
print(f"걸린 시간: {elapsed_time:.2f}s")
print(f"속도: {tokens_per_sec:.2f} tokens / sec")
