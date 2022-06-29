# The definition and usage of a function
def foo(val1, val2, val3):
    return val1 + val2 + val3

def fooo(val1, val2, val3, calcSum=True):
    # Calculate the sum; calcSum is optional with default=True
    if calcSum:
        return val1 + val2 + val3
    # Calculate the average instead
    else:
        return (val1 + val2 + val3) / 3

print(foo(3,5,7))
print(fooo(3,5,7)) # use default value for calcSum
print(fooo(3,5,7,calcSum=False))

def foooo(p1, p2, p3, n1=None, n2=None):
    print('[%d %d %d]' % (p1, p2, p3))
    if n1:
        print('n1=%d' % n1)
    if n2:
        print('n2=%d' % n2)

foooo(1, 2, 3, n2=99) # key/value pair
foooo(1, 2, n1=42, p3=3)# position and key/value pair

def lessThan(cutoffVal, *vals) : # * means any other positional parameters.
    ''' Return a list of values less than the cutoff. ''' # document in Python
    arr = []
    for val in vals :
        if val < cutoffVal:
            arr.append(val)
    return arr

print(lessThan(10, 2, 17, -3, 42, 5))

# The following function is to do something like 
# generating messages from a template that has placeholders 
# for values that will be inserted at run-time, e.g.

# Hello {name}. Your account balance is {1}, you have {2} available credit.

def formatString(stringTemplate, *args, **kwargs): 
    
    for i in range(0, len(args)):
        tmp = '{%s}' % str(1+i)
        while True:
            pos = stringTemplate.find(tmp)# find {1}, {2}... in stringTemplate
            if pos < 0:
                break
            stringTemplate = stringTemplate[:pos] + \
                             str(args[i]) + \
                             stringTemplate[pos+len(tmp):]
 
    # Replace any named parameters
    for key, val in kwargs.items():
        tmp = '{%s}' % key
        while True:
            pos = stringTemplate.find(tmp) 
            if pos < 0:
                break
            stringTemplate = stringTemplate[:pos] + \
                             str(val) + \
                             stringTemplate[pos+len(tmp):]
 
    return stringTemplate

stringTemplate = 'pos1={1} pos2={2} pos3={3} foo={foo} bar={bar}'
print(formatString(stringTemplate, 5, 9))
print(formatString(stringTemplate, 42, bar=123, foo='hello'))