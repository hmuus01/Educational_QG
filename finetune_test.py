from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("./t_t5_pre/", from_flax=True)

tokenizer = AutoTokenizer.from_pretrained('./t_t5_pre/')
input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
outputs = model.generate(input_ids)

print(outputs)