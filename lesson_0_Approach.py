# Object Oriented and procedural approach
import numpy as np # numpy is a package
# %% Object Oriented

a = np.arange(6) # a is an object that contains data and methods
b = a.reshape(2, 3) # the method act on the data
print(b.max())
print(b.max(axis=0))

# Procedural 
a = np.arange(6)
b = np.reshape(a, (2,3)) # pass data to a function
np.max(b)
print(np.max(b, axis=0))
# %% Chaining method as comaprd to the nesting function in MATLAB

sentence = "the quick brown fox "
print(sentence.strip().upper().replace(' ', '_'))

a = np.arange(12)
print(a.reshape(3,4).max(axis=0))
# nesting function, not recommended
print(np.max(np.reshape(a, (3,4)), axis=0))
# %% for loop : Do and Do not
# DO NOT
words = ['quick', 'brown', 'fox']
for i_word in range(len(words)):
    print(words[i_word]) 

# Do
words = ['quick', 'brown', 'fox']
for word in words:
    print(word) 

# DO NOT: no space and mix code
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0.1, 2 * np.pi, 41)
y=np.exp(np.sin(x))
plt.stem(x, y, linefmt='grey', markerfmt='D', bottom=1.1,label='$e^{sin(x)}$')
plt.show()

# DO: with space
x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))
plt.stem(x, y, 
            linefmt='grey', 
            markerfmt='D', 
            bottom=1.1,
            label='$e^{sin(x)}$')
plt.show()            
# %%
