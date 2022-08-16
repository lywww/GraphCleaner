import matplotlib.pyplot as plt

#Cora
#sym
# x = [1,2,3]
# f1 = [0.69, 0.71, 0.72]
# mcc = [0.68, 0.70, 0.71]
# p = [0.79, 0.81, 0.81]
# auc = [0.98, 0.98, 0.99]
#asym
x = [1,2,3]
f1 = [0.59, 0.62, 0.62]
mcc = [0.57, 0.61, 0.61]
p = [0.63, 0.67, 0.64]
auc = [0.94, 0.94, 0.94]
#CiteSeer
# x = [0,1,2,3]
# f1 = [0.37, 0.46, 0.51, 0.48]
# mcc = [0.35, 0.45, 0.50, 0.46]
# p = [0.43, 0.49, 0.49, 0.46]
# auc = [0.86, 0.91, 0.91, 0.88]
#PubMed
# x = [0,1,2,3]
# f1 = [0.44, 0.49, 0.56, 0.53]
# mcc = [0.41, 0.45, 0.53, 0.50]
# p = [0.48, 0.52, 0.64, 0.54]
# auc = [0.87, 0.89, 0.92, 0.90]
l1=plt.plot(x,f1,'r--',label='F1')
l2=plt.plot(x,mcc,'g--',label='MCC')
l3=plt.plot(x,p,'b--',label='P@|True|')
l4=plt.plot(x,auc,'y--',label='AUC')
plt.plot(x,f1,'ro-',x,mcc,'go-',x,p,'bo-',x,auc,'yo-')
plt.xticks([1,2,3])
plt.title('Complexity Experiment on Cora (asymmetric)')
plt.xlabel('row')
plt.ylabel('column')
plt.legend()
plt.savefig('Complexity_Cora_asym.jpg')
plt.show()

