import matplotlib.pyplot as plt
f = open('titanic.csv')
feature1=[]
survivedF=[]
nsurvivedF=[]
for i in f:
 try:
  t = i.split(',')
  if t[5]=='female' and int(t[1])==1:
    survivedF.append(int(t[6]))
  if t[5]=='female' and int(t[1])==0:
    nsurvivedF.append(int(t[6]))

 except:
   pass


plt.hist(survivedF, bins=20)
plt.hist(nsurvivedF, bins=20)
plt.show()
