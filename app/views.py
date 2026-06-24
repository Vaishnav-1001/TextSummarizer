from django.shortcuts import render
# Import the direct model classes instead of pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Download/Load the model directly into memory
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def home(request):
    summary = ""
    
    if request.method == 'POST':
        # Get the text from your frontend form
        raw_text = request.POST.get('input_text', '')
        
        if raw_text:
            # 2. Convert the text into numbers (tokens) the AI can understand
            inputs = tokenizer(raw_text, return_tensors="pt", max_length=1024, truncation=True)
            
            # 3. Generate the summary
            outputs = model.generate(inputs.input_ids, max_length=200, min_length=30, do_sample=False)
            
            # 4. Decode the numbers back into English text
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(summary)
            
    return render(request, 'index.html', {'summary': summary})