import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('Data.csv', header=None) #ilk satırın column isimleri olarak alınmamasını sağlar

#iki tane for loopu ile elementleri listeye ekledik
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2) # min_support = 3*7/7501
results = list(rules)
print(results)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['First Product', 'Second Product', 'Support'])
print(resultsinDataFrame)
print(resultsinDataFrame.nlargest(n=10, columns='Support'))