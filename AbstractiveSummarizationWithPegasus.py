#Installing pytorch and transformers (dependencies)
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
#pip install transformers
#Installing The Model
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

#Text to be summarized is here!
text = """ 
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.[33]

Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.[34][35]

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.[36] Python 2.0 was released in 2000 and introduced new features such as list comprehensions, cycle-detecting garbage collection, reference counting, and Unicode support. Python 3.0, released in 2008, was a major revision that is not completely backward-compatible with earlier versions. Python 2 was discontinued with version 2.7.18 in 2020.[37]

Python consistently ranks as one of the most popular programming languages.[38][39][40][41]
"""
#Create tokens - Number representation of our text

tokens = tokenizer(text, truncation = True, padding="longest", return_tensors="pt")

#print(tokens)

summary = model.generate(**tokens)

print(summary) #The Summary will be made up by tokens (numbers that represent tokens)

#Decoding Summary

print(tokenizer.decode(summary[0]))



