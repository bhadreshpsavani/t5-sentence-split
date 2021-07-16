import streamlit as st
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

@st.cache(show_spinner=False, persist=True)
def load_model(input_complex_sentence,model):
	
	base_path = "flax-community/"
	model_path = base_path + model
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
	
	tokenized_sentence = tokenizer(input_complex_sentence,return_tensors="pt")
	result = model.generate(tokenized_sentence['input_ids'],attention_mask = tokenized_sentence['attention_mask'],max_length=256,num_beams=5)
	generated_sentence = tokenizer.decode(result[0],skip_special_tokens=True)
	
	return generated_sentence

def main():

	st.sidebar.title("🧠Sentence Simplifier")
	st.title("Sentence Split in English using T5 Variants")
	st.write("Sentence Split is the task of **dividing a long Complex Sentence into Simple Sentences**")
	
	model = st.sidebar.selectbox(
				  "Please Choose the Model",
				   ("t5-base-wikisplit","t5-v1_1-base-wikisplit", "byt5-base-wikisplit","t5-large-wikisplit"))

	st.sidebar.write('''
		## Applications:
		* Sentence Simplification
		* Data Augmentation
		* Sentence Rephrase
	''')

	st.sidebar.write("[More Exploration](https://github.com/bhadreshpsavani/t5-sentence-split)")
	
	example = "Mary likes to play football in her freetime whenever she meets with her friends that are very nice people."
	input_complex_sentence = st.text_area("Please type a Complex Sentence to split",example)

	if st.button('Split✂️'):
		with st.spinner("Spliting Sentence...🧠"):
			generated_sentence = load_model(input_complex_sentence, model)
		st.write(generated_sentence)


if __name__ == "__main__":
	main()
