from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
from configs import MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
).cuda().eval()


# synth,cot composite — reasoning / CoT style (see Disclaimer for other modes)
query = "9.8 and 9.11, which is bigger?"
condition = "<|quad_end|><|object_ref_end|>"
prompt = f"<|im_start|>{condition}{query}<|im_end|>"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# Mark the prompt as a single bidirectional prefix block — see "PrefixLM mask" below.
inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=False,
)

gen_kwargs = dict(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    streamer=streamer,
)

# Run generation in a background thread; main thread drains the streamer.
thread = Thread(target=model.generate, kwargs=gen_kwargs)
with torch.no_grad():
    thread.start()
    for new_text in streamer:
        print(new_text, end="", flush=True)
    thread.join()
print() 