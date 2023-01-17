pip install mlxtend
import pandas
from mlxtend.preprocessing import TransactionEncoder
#read data
dinosaurList = pandas.read_csv('dinosaur.csv')
#print data
display(dinosaurList)
"""
#Itemset generation for finding association between the period and the diet of the dinosaurs.
"""

# Normalize, strip and lower case the values of Period and Diet columns

dinosaurList["Period"] = dinosaurList["Period"].str.normalize('NFKD')
dinosaurList["Diet"] = dinosaurList["Diet"].str.normalize('NFKD')
dinosaurList["Period"] = dinosaurList["Period"].str.strip()
dinosaurList["Diet"] = dinosaurList["Diet"].str.strip()
dinosaurList["Period"] = dinosaurList["Period"].str.replace('(','').replace(')','')
dinosaurList["Diet"] = dinosaurList["Diet"].str.replace('(','')
dinosaurList["Diet"] = dinosaurList["Diet"].str.replace(')','')
dinosaurList["Diet"] = dinosaurList["Diet"].str.replace('?','')
dinosaurList["Period"] = dinosaurList["Period"].str.lower()
dinosaurList["Diet"] = dinosaurList["Diet"].str.lower() 

# Creating a new dataframe with data from Period and Diet columns
dataa = [list(dList) for dList in zip(dinosaurList["Period"], dinosaurList["Diet"])]
#Using transaction encoder to transform the dataframe
trnxnEncdr = TransactionEncoder()
trnxnEncdrArry = trnxnEncdr.fit(dataa).transform(dataa)
encodedDF = pandas.DataFrame(trnxnEncdrArry, columns=trnxnEncdr.columns_)
encodedDF
from mlxtend.frequent_patterns import apriori

#Minimum support in this experiment = 10%
#Minimum confidence = 10%
# Generating frequent itemsets
frqIt = apriori(encodedDF, min_support=0.1, use_colnames=True)
frqIt['length'] = frqIt['itemsets'].apply(lambda x: len(x))
frqIt
# Filtering frequent itemsets of length 2 and support >= 10%
frqIt[ (frqIt['length'] == 2) & (frqIt['support'] >= 0.1) ]
"""
#Rule Generation using specified confidence value
"""
from mlxtend.frequent_patterns import association_rules

assRules = association_rules(frqIt, metric="confidence", min_threshold=0.1)

assRules
