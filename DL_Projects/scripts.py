s=['michael','michael','alice','carter']

mydict={}
i = 0
for item in s:
    if(i>0 and item in mydict):
        continue
    else:
        i = i+1
        mydict[item] = i

k=[]
for item in s:
    k.append(mydict[item])

print(k)