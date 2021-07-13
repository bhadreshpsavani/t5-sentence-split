import streamlit as st
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

@st.cache(persist=True)
def load_model(input_complex_sentence,model):
	
	base_path = "flax-community/"
	model_path = base_path + model
	print(model_path)
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
	
	tokenized_sentence = tokenizer(input_complex_sentence,return_tensors="pt")
	result = model.generate(tokenized_sentence['input_ids'],attention_mask = tokenized_sentence['attention_mask'],max_length=256,num_beams=5)
	print(result)
	generated_sentence = tokenizer.decode(result[0],skip_special_tokens=True)
	
	return generated_sentence

def main():

	st.title("Sentence Split in English using T5 variants")
	st.write("Sentence Split is the task of dividing a long Sentence into multiple Sentences")
	
	model = st.sidebar.selectbox(
				  "Please Choose the Model",
				   ("t5-base-wikisplit","t5-large-wikisplit"))
	st.write("Model Selected : ", model)
	
	example = "Mary likes to play football in her freetime whenever she meets with her friends that are very nice people."
	input_complex_sentence = st.text_area("Please type a long Sentence to split",example)

	if st.button('Split it!'):
 
		generated_sentence = load_model(input_complex_sentence,model)
		st.write(generated_sentence)


if __name__ == "__main__":
	main()